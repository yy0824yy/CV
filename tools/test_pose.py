"""
Step 5 验证脚本：MediaPipe Pose 接入测试。

效果：
- 实时显示 RGB 画面，叠加 33 关键点人体骨架
- 主要关节（肩肘腕髋膝踝）用红色大点突出
- 可见度低的点用暗色显示
- 左上角显示 FPS + 关键点可见数
- 同时叠加显示手部识别（Step 3/4 已完成），便于看整体效果

用法：
    python tools/test_pose.py
    python tools/test_pose.py --no-hand   # 只显示姿态，不跑手部
按 q 退出, s 截图, h 切换手部检测显示
"""
import argparse
import os
import sys
import time

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.camera import create_camera
from core.hand_detector import HandDetector
from core.gesture_recognizer import GestureRecognizer
from core.pose_detector import PoseDetector
from config import CAMERA, DISPLAY


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source", default=CAMERA.source)
    p.add_argument("--no-hand", action="store_true", help="不跑手部检测，仅姿态")
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
    hand_det = HandDetector(max_num_hands=2, already_flipped=False)
    recognizer = GestureRecognizer()

    enable_hand = not args.no_hand
    fps, prev_t = 0.0, time.time()

    print(f"[INFO] 源={args.source}, hand_enabled={enable_hand}")
    print("[INFO] 按 q 退出, s 截图, h 切换手部")

    with cam:
        while True:
            frame = cam.read()
            if frame is None:
                continue
            color = frame.color
            if flip:
                color = cv2.flip(color, 1)

            # 姿态检测
            pose = pose_det.detect(color)
            pose_det.draw(color, pose)

            # 手部（可选）
            if enable_hand:
                hands = hand_det.detect(color)
                results = recognizer.recognize_all(hands)
                hand_det.draw(color, hands)
                for hand, res in zip(hands, results):
                    wrist = tuple(hand.landmarks_px[0])
                    cv2.putText(color, res.name,
                                (wrist[0] - 40, wrist[1] + 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 215, 255), 2)

            # FPS
            now = time.time()
            inst_fps = 1.0 / max(now - prev_t, 1e-6)
            fps = fps * 0.9 + inst_fps * 0.1 if fps > 0 else inst_fps
            prev_t = now

            # 顶部信息
            cv2.putText(color, f"FPS: {fps:.1f}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            n_vis = int((pose.visibility > 0.5).sum()) if pose is not None else 0
            cv2.putText(color, f"Pose: {'YES' if pose else 'NO'}  visible={n_vis}/33",
                        (20, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(color, f"Hand: {'ON' if enable_hand else 'OFF'}",
                        (20, 105),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            cv2.imshow("Pose + Hand (q=quit, s=save, h=toggle hand)", color)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                ts = int(time.time())
                cv2.imwrite(f"pose_snap_{ts}.png", color)
                print(f"[INFO] saved pose_snap_{ts}.png")
            elif key == ord('h'):
                enable_hand = not enable_hand

    pose_det.close()
    hand_det.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
