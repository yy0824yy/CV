"""
Step 3 验证脚本：MediaPipe Hands 接入测试。

效果：
- 实时显示 RGB 画面
- 检测到的手画出 21 关键点 + 连接骨架
- 指尖用红色大点突出
- 标注左右手 + 置信度
- 左上角显示检测到的手数 + FPS

用法：
    python tools/test_hand_detection.py
    python tools/test_hand_detection.py --source opencv:0
按 q 退出，按 s 截图。
"""
import argparse
import os
import sys
import time

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.camera import create_camera
from core.hand_detector import HandDetector
from config import CAMERA, DISPLAY


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
    # 因为我们前面会做 cv2.flip 镜像，所以 detector 也要告诉它已经翻转过了
    detector = HandDetector(
        max_num_hands=2,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
        already_flipped=False,  # MediaPipe 0.10.x 在镜像输入下标签已正确，不再对调
    )

    fps = 0.0
    prev_t = time.time()

    print(f"[INFO] 摄像头源: {args.source}")
    print("[INFO] 按 q 退出, s 截图")

    with cam:
        while True:
            frame = cam.read()
            if frame is None:
                continue

            color = frame.color
            if flip:
                color = cv2.flip(color, 1)

            # 检测手部
            hands = detector.detect(color)
            # 在画面上画出关键点
            detector.draw(color, hands)

            # FPS
            now = time.time()
            inst_fps = 1.0 / max(now - prev_t, 1e-6)
            fps = fps * 0.9 + inst_fps * 0.1 if fps > 0 else inst_fps
            prev_t = now

            cv2.putText(color, f"FPS: {fps:.1f}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.putText(color, f"Hands: {len(hands)}", (20, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # 在画面右上角列出每只手的简要信息
            for i, h in enumerate(hands):
                wrist_px = h.landmarks_px[0]
                index_tip = h.landmarks_px[8]
                cv2.putText(color,
                            f"{h.handedness}: wrist=({wrist_px[0]},{wrist_px[1]}) "
                            f"index_tip=({index_tip[0]},{index_tip[1]})",
                            (20, 110 + i * 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            cv2.imshow("Hand Detection Test (q=quit, s=save)", color)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                ts = int(time.time())
                cv2.imwrite(f"hand_snap_{ts}.png", color)
                print(f"[INFO] 已保存截图: hand_snap_{ts}.png")

    detector.close()
    cv2.destroyAllWindows()
    print("[INFO] 已退出")


if __name__ == "__main__":
    main()
