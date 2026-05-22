"""
Step 2 验证脚本：测试摄像头抽象层。

用法：
    # 使用 Azure Kinect（默认）
    python tools/test_camera_layer.py

    # 使用 OpenCV 0 号摄像头
    python tools/test_camera_layer.py --source opencv:0

操作：
    q   退出
    s   截图（彩色 + 深度 + 原始深度 .npy）
    f   开关镜像翻转
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np

# 让 tools 脚本能 import 项目根目录下的 core / config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.camera import create_camera
from core.depth_utils import colorize_depth, get_distance_mm, resize_depth_to_color
from config import CAMERA, DISPLAY


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source", default=CAMERA.source,
                   help="摄像头源，'kinect' 或 'opencv:0'")
    return p.parse_args()


def main():
    args = parse_args()

    # 根据 source 创建摄像头（工厂模式）
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

    print(f"[INFO] 摄像头源: {args.source}")
    print(f"[INFO] 支持深度: {cam.has_depth()}")
    print("[INFO] 正在打开设备 ...")

    flip = DISPLAY.flip_horizontal
    fps = 0.0
    prev_t = time.time()

    with cam:  # 自动 open/close
        print("[INFO] 设备已打开。按 q 退出，s 截图，f 翻转。")
        while True:
            frame = cam.read()
            if frame is None:
                continue

            color = frame.color
            depth = frame.depth

            # 镜像翻转
            if flip:
                color = cv2.flip(color, 1)
                if depth is not None:
                    depth = cv2.flip(depth, 1)

            # FPS 平滑
            now = time.time()
            inst_fps = 1.0 / max(now - prev_t, 1e-6)
            fps = fps * 0.9 + inst_fps * 0.1 if fps > 0 else inst_fps
            prev_t = now

            # 中心点距离（仅 Kinect）
            center_text = ""
            if depth is not None:
                h, w = depth.shape[:2]
                d_mm = get_distance_mm(depth, w // 2, h // 2, window=11)
                if d_mm is not None:
                    center_text = f"Center: {d_mm/1000:.2f} m"

            # 叠加文字
            cv2.putText(color, f"FPS: {fps:.1f}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.putText(color, f"Source: {args.source}", (20, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            if center_text:
                cv2.putText(color, center_text, (20, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # 显示画面
            if depth is not None:
                # 把深度图缩放到与彩色一致，再做伪彩拼接显示
                depth_resized = resize_depth_to_color(depth, color.shape)
                depth_color = colorize_depth(depth_resized, max_mm=DISPLAY.depth_max_mm)
                # 在深度伪彩中心标记一个十字
                hc, wc = depth_color.shape[:2]
                cv2.drawMarker(depth_color, (wc // 2, hc // 2), (255, 255, 255),
                               cv2.MARKER_CROSS, 30, 2)
                combined = np.hstack([color, depth_color])
                cv2.imshow("Camera Layer Test (q=quit, s=save, f=flip)", combined)
            else:
                cv2.imshow("Camera Layer Test (q=quit, s=save, f=flip)", color)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('f'):
                flip = not flip
                print(f"[INFO] 镜像翻转: {flip}")
            elif key == ord('s'):
                ts = int(time.time())
                cv2.imwrite(f"snap_color_{ts}.png", color)
                if depth is not None:
                    cv2.imwrite(f"snap_depth_{ts}.png", depth_color)
                    np.save(f"snap_depth_raw_{ts}.npy", depth)
                print(f"[INFO] 已保存截图: snap_*_{ts}.*")

    cv2.destroyAllWindows()
    print("[INFO] 已退出")


if __name__ == "__main__":
    main()
