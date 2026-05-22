"""
Step 6 验证脚本：动作识别测试。

效果：
- 实时检测姿态 + 识别动作
- 画面右上角大字显示主动作（primary）
- 左下角列出所有同时满足的标签 + 调试特征值

测试动作：
    举左手 / 举右手 / 举双手
    左倾 / 右倾
    弯腰 (向前弯 30° 以上)
    下蹲 (膝盖弯到接近髋部)
    站直 (Standing)

按 q 退出, s 截图, d 切换调试信息
"""
import argparse
import os
import sys
import time

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.camera import create_camera
from core.pose_detector import PoseDetector
from core.action_recognizer import ActionRecognizer
from config import CAMERA, DISPLAY


# 每种动作的显示颜色
ACTION_COLORS = {
    "Raise_Left_Hand":   (0, 255, 255),
    "Raise_Right_Hand":  (255, 255, 0),
    "Raise_Both_Hands":  (0, 255, 0),
    "Squat":             (0, 165, 255),
    "Bend_Over":         (255, 0, 255),
    "Lean_Left":         (255, 100, 100),
    "Lean_Right":        (100, 100, 255),
    "Standing":          (200, 200, 200),
    "Unknown":           (128, 128, 128),
    "No_Person":         (80, 80, 80),
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source", default=CAMERA.source)
    return p.parse_args()


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
    action_rec = ActionRecognizer()

    fps, prev_t = 0.0, time.time()
    show_debug = True

    print(f"[INFO] 源={args.source}")
    print("[INFO] 按 q 退出, s 截图, d 切换调试")

    with cam:
        while True:
            frame = cam.read()
            if frame is None:
                continue
            color = frame.color
            if flip:
                color = cv2.flip(color, 1)

            pose = pose_det.detect(color)
            pose_det.draw(color, pose)
            result = action_rec.recognize(pose)

            # FPS
            now = time.time()
            inst_fps = 1.0 / max(now - prev_t, 1e-6)
            fps = fps * 0.9 + inst_fps * 0.1 if fps > 0 else inst_fps
            prev_t = now

            # 顶部状态
            cv2.putText(color, f"FPS: {fps:.1f}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            # 主动作（右上角大字）
            h, w = color.shape[:2]
            col = ACTION_COLORS.get(result.primary, (255, 255, 255))
            cv2.putText(color, result.primary,
                        (w - 500, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, col, 3)

            # 所有标签
            if result.labels and result.labels != [result.primary]:
                cv2.putText(color, "Labels: " + ", ".join(result.labels),
                            (w - 500, 85),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            # 调试特征（左下角）
            if show_debug and result.valid:
                y = h - 20
                for k, v in result.features.items():
                    cv2.putText(color, f"{k} = {v:.3f}", (20, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    y -= 22

            cv2.imshow("Action Recognition (q=quit, s=save, d=debug)", color)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                ts = int(time.time())
                cv2.imwrite(f"action_snap_{ts}.png", color)
                print(f"[INFO] saved action_snap_{ts}.png")
            elif key == ord('d'):
                show_debug = not show_debug

    pose_det.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
