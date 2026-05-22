"""
Step 7 验证脚本：关节角度计算 + 滑动平均平滑。

效果：
- 实时计算 6 个关节角度（左/右肘、左/右膝、左/右肩）
- 在每个关节附近显示角度数字
- 同时显示 平滑前 vs 平滑后 的肘关节角度（折线图）
- 演示平滑对抖动的抑制效果

按 q 退出, s 截图, p 暂停/继续, r 重置平滑器
"""
import argparse
import os
import sys
import time
from collections import deque

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.camera import create_camera
from core.pose_detector import (
    PoseDetector,
    LEFT_ELBOW, RIGHT_ELBOW,
    LEFT_KNEE, RIGHT_KNEE,
    LEFT_SHOULDER, RIGHT_SHOULDER,
)
from core.geometry import compute_body_angles
from core.smoothing import EMASmoother, LandmarkSmoother
from config import CAMERA, DISPLAY


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source", default=CAMERA.source)
    p.add_argument("--alpha", type=float, default=0.35,
                   help="EMA alpha for angles (smaller = smoother)")
    p.add_argument("--lm-alpha", type=float, default=0.7,
                   help="EMA alpha for landmark coords")
    return p.parse_args()


def draw_angle(img, p_join, angle_value, label, color=(0, 255, 255)):
    """在关节点附近标注角度值。"""
    if angle_value is None:
        return
    x, y = int(p_join[0]), int(p_join[1])
    txt = f"{label}:{angle_value:.0f}"
    cv2.putText(img, txt, (x + 8, y + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def draw_curve(img, raw_buf, smooth_buf, x0, y0, w, h, label):
    """在画面右下角画一对折线，对比平滑前后。"""
    cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), (60, 60, 60), 1)
    cv2.putText(img, label, (x0 + 4, y0 + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    if len(raw_buf) < 2:
        return
    n = len(raw_buf)
    # 角度范围 0~180 映射到 y
    def to_xy(buf, color):
        pts = []
        for i, v in enumerate(buf):
            x = x0 + int(i * w / max(n - 1, 1))
            y = y0 + h - int((v / 180.0) * (h - 16)) - 4
            pts.append((x, y))
        for i in range(1, len(pts)):
            cv2.line(img, pts[i - 1], pts[i], color, 1)
    to_xy(raw_buf, (80, 80, 255))     # 红：原始
    to_xy(smooth_buf, (80, 255, 80))  # 绿：平滑
    cv2.putText(img, "raw", (x0 + 4, y0 + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 80, 255), 1)
    cv2.putText(img, "smoothed", (x0 + 40, y0 + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 255, 80), 1)


def main():
    args = parse_args()
    if args.source == "kinect":
        cam = create_camera(
            "kinect",
            color_resolution=CAMERA.kinect_color_resolution,
            depth_mode=CAMERA.kinect_depth_mode,
            fps=CAMERA.kinect_fps,
        )
    else:
        cam = create_camera(
            args.source,
            width=CAMERA.opencv_width,
            height=CAMERA.opencv_height,
        )

    flip = DISPLAY.flip_horizontal
    pose_det = PoseDetector(model_complexity=1)

    # 平滑器：关键点平滑 + 6 个角度的独立 EMA
    lm_smoother = LandmarkSmoother(alpha=args.lm_alpha)
    angle_smoothers = {
        k: EMASmoother(alpha=args.alpha)
        for k in ["left_elbow", "right_elbow",
                  "left_knee", "right_knee",
                  "left_shoulder", "right_shoulder",
                  "shoulder_tilt", "torso_tilt"]
    }

    # 折线图缓冲（用左肘做演示）
    BUF_LEN = 120
    raw_buf = deque(maxlen=BUF_LEN)
    smooth_buf = deque(maxlen=BUF_LEN)

    paused = False
    fps, prev_t = 0.0, time.time()

    print(f"[INFO] 源={args.source}, alpha={args.alpha}, lm_alpha={args.lm_alpha}")
    print("[INFO] 按 q 退出, s 截图, p 暂停, r 重置平滑器")

    with cam:
        while True:
            if not paused:
                frame = cam.read()
                if frame is None:
                    continue
                color = frame.color
                if flip:
                    color = cv2.flip(color, 1)

                pose = pose_det.detect(color)
                if pose is not None:
                    h, w = color.shape[:2]

                    # ---- 平滑关键点（归一化坐标）----
                    smoothed_norm = lm_smoother.update(pose.landmarks_norm)
                    smoothed_px = np.zeros((33, 2), dtype=np.int32)
                    smoothed_px[:, 0] = np.clip(smoothed_norm[:, 0] * w, 0, w - 1)
                    smoothed_px[:, 1] = np.clip(smoothed_norm[:, 1] * h, 0, h - 1)

                    # ---- 角度计算（基于原始 vs 平滑） ----
                    raw_angles = compute_body_angles(pose.landmarks_norm)
                    smooth_input_angles = compute_body_angles(smoothed_norm)
                    final_angles = {
                        k: angle_smoothers[k].update(v)
                        for k, v in smooth_input_angles.items()
                    }

                    # ---- 绘制骨架（用平滑后的像素坐标）----
                    pose.landmarks_px = smoothed_px  # 临时替换为平滑后的
                    pose_det.draw(color, pose)

                    # ---- 在关节附近标注角度 ----
                    draw_angle(color, smoothed_px[LEFT_ELBOW], final_angles["left_elbow"], "LE")
                    draw_angle(color, smoothed_px[RIGHT_ELBOW], final_angles["right_elbow"], "RE")
                    draw_angle(color, smoothed_px[LEFT_KNEE], final_angles["left_knee"], "LK")
                    draw_angle(color, smoothed_px[RIGHT_KNEE], final_angles["right_knee"], "RK")
                    draw_angle(color, smoothed_px[LEFT_SHOULDER], final_angles["left_shoulder"], "LS")
                    draw_angle(color, smoothed_px[RIGHT_SHOULDER], final_angles["right_shoulder"], "RS")

                    # ---- 折线图缓冲：左肘原始 vs 平滑后 ----
                    raw_buf.append(raw_angles["left_elbow"])
                    smooth_buf.append(final_angles["left_elbow"])

                # ---- FPS ----
                now = time.time()
                inst_fps = 1.0 / max(now - prev_t, 1e-6)
                fps = fps * 0.9 + inst_fps * 0.1 if fps > 0 else inst_fps
                prev_t = now

                cv2.putText(color, f"FPS: {fps:.1f}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                cv2.putText(color, f"alpha={args.alpha}  lm_alpha={args.lm_alpha}",
                            (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

                # 折线图（右下）
                hh, ww = color.shape[:2]
                draw_curve(color, raw_buf, smooth_buf,
                           x0=ww - 320, y0=hh - 120, w=300, h=100,
                           label="Left Elbow Angle (raw vs smoothed)")

                last_frame = color

            cv2.imshow("Angles + Smoothing (q=quit, s=save, p=pause, r=reset)", last_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                ts = int(time.time())
                cv2.imwrite(f"angle_snap_{ts}.png", last_frame)
                print(f"[INFO] saved angle_snap_{ts}.png")
            elif key == ord('p'):
                paused = not paused
            elif key == ord('r'):
                lm_smoother.reset()
                for s in angle_smoothers.values():
                    s.reset()
                raw_buf.clear()
                smooth_buf.clear()
                print("[INFO] smoothers reset")

    pose_det.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
