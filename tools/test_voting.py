"""
Step 8 验证脚本：连续帧多数投票稳定化。

并排显示同一只手的两个识别结果：
    左侧：raw（原始单帧识别）
    右侧：voted（经过 7 帧多数投票）

观察：故意快速变换手势或在两个手势之间犹豫时，raw 会跳变，voted 更稳定。

按 q 退出, s 截图
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
from core.stable_recognizers import StableGestureRecognizer
from config import CAMERA, DISPLAY


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source", default=CAMERA.source)
    p.add_argument("--window", type=int, default=7)
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
    detector = HandDetector(max_num_hands=2, already_flipped=False)
    raw_rec = GestureRecognizer()
    stable_rec = StableGestureRecognizer(window=args.window)

    fps, prev_t = 0.0, time.time()

    print(f"[INFO] 源={args.source}, voting window={args.window}")
    print("[INFO] 按 q 退出, s 截图")

    with cam:
        while True:
            frame = cam.read()
            if frame is None:
                continue
            color = frame.color
            if flip:
                color = cv2.flip(color, 1)

            hands = detector.detect(color)
            raw_results = raw_rec.recognize_all(hands)
            voted_results = stable_rec.recognize_all(hands)
            detector.draw(color, hands)

            # 在每只手腕处标 raw / voted 两个结果
            for hand, raw, voted in zip(hands, raw_results, voted_results):
                wx, wy = hand.landmarks_px[0]
                # 不一致时高亮：raw 红 / voted 绿
                if raw.name != voted.name:
                    raw_col = (60, 60, 255)
                    voted_col = (60, 255, 60)
                else:
                    raw_col = (180, 180, 180)
                    voted_col = (0, 215, 255)
                cv2.putText(color, f"raw  : {raw.name}",
                            (wx - 80, wy + 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, raw_col, 2)
                cv2.putText(color, f"voted: {voted.name}",
                            (wx - 80, wy + 75),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, voted_col, 2)

            now = time.time()
            inst_fps = 1.0 / max(now - prev_t, 1e-6)
            fps = fps * 0.9 + inst_fps * 0.1 if fps > 0 else inst_fps
            prev_t = now

            cv2.putText(color, f"FPS: {fps:.1f}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.putText(color, f"voting window = {args.window} frames",
                        (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(color, "raw vs voted (red=mismatch)",
                        (20, 92),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            cv2.imshow("Voting Stabilization (q=quit, s=save)", color)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                ts = int(time.time())
                cv2.imwrite(f"voting_snap_{ts}.png", color)
                print(f"[INFO] saved voting_snap_{ts}.png")

    detector.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
