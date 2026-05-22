"""
Azure Kinect Body Tracking SDK 独立测试脚本。

目的：验证 SDK 能在 Python 中工作并实时检测 3D 人体骨架。
依赖：pykinect_azure（pip install pykinect_azure）
     + Body Tracking SDK 已装好且 PATH 已配置
     + Kinect 设备已连接

操作：
    q  退出
    s  截图（保存到 tools/output_body_tracking.png）

注意：
    Body Tracking 需要 NVIDIA GPU（默认 CUDA 后端）。
    如果跑不动可以改 processing_mode 为 CPU 或 DirectML（见下方）。

输出关键点（K4ABT 32 关节）：
    Pelvis / Spine_Navel / Spine_Chest / Neck / 
    Clavicle_Left/Right / Shoulder_Left/Right / Elbow_Left/Right /
    Wrist_Left/Right / Hand_Left/Right / Handtip_Left/Right / Thumb_Left/Right /
    Hip_Left/Right / Knee_Left/Right / Ankle_Left/Right / Foot_Left/Right /
    Head / Nose / Eye_Left/Right / Ear_Left/Right
"""
from __future__ import annotations

import os
import sys
import time

import cv2
import numpy as np

# ============================================================
# 1. 初始化 SDK
# ============================================================
import pykinect_azure as pykinect

pykinect.initialize_libraries(track_body=True)


# ============================================================
# 2. 配置：颜色 + 深度 + Body Tracking
# ============================================================
device_config = pykinect.default_configuration
device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_720P
# 深度模式：NFOV_UNBINNED 是 Body Tracking 的推荐模式
# 视野较窄但精度高；如果想要更广视野可换 WFOV_2X2BINNED
device_config.depth_mode = pykinect.K4A_DEPTH_MODE_NFOV_UNBINNED
device_config.camera_fps = pykinect.K4A_FRAMES_PER_SECOND_15
device_config.synchronized_images_only = True

print("[INFO] 配置:")
print(f"  彩色: 720P")
print(f"  深度: NFOV_UNBINNED")
print(f"  帧率: 15 FPS")

# ============================================================
# 3. 启动设备 + Body Tracker
# ============================================================
print("[INFO] 启动 Kinect 设备 ...")
device = pykinect.start_device(config=device_config)

print("[INFO] 启动 Body Tracker（首次启动 GPU 推理引擎可能需要 5~15 秒）...")
bodyTracker = pykinect.start_body_tracker()
print("[INFO] Body Tracker OK")

print("[INFO] 操作: q 退出 | s 截图")


# ============================================================
# 4. 主循环
# ============================================================
last_t = time.time()
frame_count = 0
fps = 0.0
last_color = None

try:
    while True:
        # 取一帧 capture
        capture = device.update()
        # 用最新的 capture 喂给 body tracker
        body_frame = bodyTracker.update()

        # 拿彩色图
        ret_color, color_image = capture.get_color_image()
        if not ret_color:
            continue

        # 拿深度伪彩图（可选，用于小窗显示）
        ret_depth, depth_color = capture.get_colored_depth_image()

        # 拷一份用来绘制
        display = color_image.copy()
        # color_image 是 BGRA，转 BGR 便于 OpenCV 显示
        if display.shape[-1] == 4:
            display = cv2.cvtColor(display, cv2.COLOR_BGRA2BGR)

        # 把 body 关键点画到彩色画面上
        # pykinect_azure 提供 body_frame.draw_bodies(image, ...) 直接绘制
        try:
            display = body_frame.draw_bodies(display, pykinect.K4A_CALIBRATION_TYPE_COLOR)
        except Exception as e:
            # 如果 draw_bodies 不可用，回退到手动绘制
            num_bodies = body_frame.get_num_bodies()
            for b_idx in range(num_bodies):
                body = body_frame.get_body(b_idx)
                # 获取 2D 投影到彩色相机
                try:
                    body2d = body.numpy()  # shape ~ (32, 4) [x, y, conf, ?]
                    for j in range(body2d.shape[0]):
                        x, y = int(body2d[j, 0]), int(body2d[j, 1])
                        if 0 <= x < display.shape[1] and 0 <= y < display.shape[0]:
                            cv2.circle(display, (x, y), 4, (0, 255, 0), -1)
                except Exception:
                    pass

        # FPS 计算
        frame_count += 1
        now = time.time()
        if now - last_t >= 0.5:
            fps = frame_count / (now - last_t)
            frame_count = 0
            last_t = now

        # 读人数
        num_bodies = body_frame.get_num_bodies()

        # 顶部信息条
        cv2.rectangle(display, (0, 0), (display.shape[1], 36),
                      (40, 40, 40), -1)
        cv2.putText(display, f"FPS: {fps:.1f}  Bodies: {num_bodies}  "
                             f"[Body Tracking SDK]",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 255), 2)

        # 显示
        # 如果原图 1280x720 太大，缩小一半
        if display.shape[1] > 1280:
            display = cv2.resize(display, (1280, 720))
        cv2.imshow("Body Tracking SDK Test", display)

        # 深度小窗（可选）
        if ret_depth and depth_color is not None:
            depth_show = cv2.resize(depth_color,
                                    (depth_color.shape[1] // 2,
                                     depth_color.shape[0] // 2))
            cv2.imshow("Depth (colored)", depth_show)

        last_color = display

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s") and last_color is not None:
            out_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "output_body_tracking.png",
            )
            cv2.imwrite(out_path, last_color)
            print(f"[SHOT] {out_path}")

finally:
    try:
        device.close()
    except Exception:
        pass
    cv2.destroyAllWindows()
    print("[INFO] 已关闭")
