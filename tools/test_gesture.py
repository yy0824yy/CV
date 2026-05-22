"""
Step 4 验证脚本：手势识别测试。

效果：
- 实时检测手部 + 识别 8 种手势
- 在每只手腕处显示手势名 + 置信度
- 屏幕左上角显示调试信息（手指伸直状态、拇-食距离等）

测试这些手势：
    数字 1 / 2 / 3 / 5（张开手掌）
    握拳 Fist
    OK
    点赞 Thumbs_Up

按 q 退出, s 截图, d 切换调试信息显示
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
from config import CAMERA, DISPLAY


# 每种手势用不同颜色显示，方便区分
GESTURE_COLORS = {
    "Number_1":   (255, 255, 0),
    "Number_2":   (0, 255, 255),
    "Number_3":   (255, 0, 255),
    "Open_Palm":  (0, 255, 0),
    "Fist":       (0, 0, 255),
    "OK":         (255, 165, 0),
    "Thumbs_Up":  (0, 215, 255),
    "Unknown":    (128, 128, 128),
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
    detector = HandDetector(max_num_hands=2, already_flipped=False)
    recognizer = GestureRecognizer(min_score=0.6)

    fps, prev_t = 0.0, time.time()
    show_debug = True

    print(f"[INFO] 摄像头源: {args.source}")
    print("[INFO] 按 q 退出, s 截图, d 切换调试信息")

    with cam:
        while True:
            frame = cam.read()
            if frame is None:
                continue
            color = frame.color
            if flip:
                color = cv2.flip(color, 1)

            hands = detector.detect(color)
            results = recognizer.recognize_all(hands)
            detector.draw(color, hands)

            # FPS
            now = time.time()
            inst_fps = 1.0 / max(now - prev_t, 1e-6)
            fps = fps * 0.9 + inst_fps * 0.1 if fps > 0 else inst_fps
            prev_t = now

            cv2.putText(color, f"FPS: {fps:.1f}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            # 在每只手上显示手势名
            for hand, res in zip(hands, results):
                wrist = tuple(hand.landmarks_px[0])
                col = GESTURE_COLORS.get(res.name, (200, 200, 200))
                txt = f"{res.name} ({res.score:.2f})"
                # 加粗显示手势名
                cv2.putText(color, txt,
                            (wrist[0] - 60, wrist[1] + 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 3)

            # 调试信息：左下角列出每只手的细节
            if show_debug:
                y = color.shape[0] - 20
                for i, (hand, res) in enumerate(zip(hands[::-1], results[::-1])):
                    fs = "".join(["1" if s else "0" for s in res.finger_states])
                    line = (
                        f"[{hand.handedness}] gesture={res.name:10s} "
                        f"fingers={fs} "
                        f"thumb_idx_dist={res.extra.get('thumb_index_dist',0):.2f} "
                        f"thumb_up={res.extra.get('thumb_up_ratio',0):.2f}"
                    )
                    cv2.putText(color, line, (20, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    y -= 25

            cv2.imshow("Gesture Recognition (q=quit, s=save, d=debug)", color)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                ts = int(time.time())
                cv2.imwrite(f"gesture_snap_{ts}.png", color)
                print(f"[INFO] saved gesture_snap_{ts}.png")
            elif key == ord('d'):
                show_debug = not show_debug

    detector.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
