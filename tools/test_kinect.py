"""
最小验证脚本 2：用 pyk4a 打开 Azure Kinect DK，同时显示 RGB 和深度伪彩图。
按 q 退出，按 s 同时保存 RGB + Depth 截图。

用途：确认 Sensor SDK + pyk4a 工作正常，Kinect 设备能正常出 RGB 和 Depth 数据。
"""
import cv2
import numpy as np
import time

try:
    from pyk4a import PyK4A, Config, ColorResolution, DepthMode, FPS
except ImportError as e:
    print("[ERROR] 无法导入 pyk4a，请先 pip install pyk4a")
    raise


def colorize_depth(depth: np.ndarray, max_mm: int = 4000) -> np.ndarray:
    """将 16 位深度图（单位 mm）转换为 8 位伪彩色图，便于可视化。"""
    # 限制范围并归一化到 0-255
    d = np.clip(depth, 0, max_mm).astype(np.float32)
    d = (d / max_mm * 255.0).astype(np.uint8)
    # JET 伪彩：近=红，远=蓝
    return cv2.applyColorMap(d, cv2.COLORMAP_JET)


def main():
    # 配置 Kinect：彩色 720p BGRA + 深度 NFOV unbinned + 30 FPS
    k4a = PyK4A(Config(
        color_resolution=ColorResolution.RES_720P,
        depth_mode=DepthMode.NFOV_UNBINNED,
        camera_fps=FPS.FPS_30,
        synchronized_images_only=True,  # 只输出 RGB 和 Depth 都齐的帧
    ))

    print("[INFO] 正在打开 Azure Kinect DK ...")
    k4a.start()
    print("[INFO] Kinect 已启动。按 q 退出，按 s 截图。")

    prev_t = time.time()
    fps = 0.0

    try:
        while True:
            capture = k4a.get_capture()
            if capture.color is None or capture.depth is None:
                continue

            # 彩色：BGRA -> BGR
            color_bgr = cv2.cvtColor(capture.color, cv2.COLOR_BGRA2BGR)
            # 深度：mm 单位的 16 位 → 伪彩
            depth_color = colorize_depth(capture.depth, max_mm=4000)

            # 把深度图缩放到与彩色图同高，便于横向拼接显示
            h_color = color_bgr.shape[0]
            scale = h_color / depth_color.shape[0]
            depth_resized = cv2.resize(
                depth_color,
                (int(depth_color.shape[1] * scale), h_color)
            )

            # FPS
            now = time.time()
            inst_fps = 1.0 / max(now - prev_t, 1e-6)
            fps = fps * 0.9 + inst_fps * 0.1 if fps > 0 else inst_fps
            prev_t = now
            cv2.putText(color_bgr, f"FPS: {fps:.1f}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            combined = np.hstack([color_bgr, depth_resized])
            cv2.imshow("Kinect: RGB | Depth (q=quit, s=save)", combined)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                ts = int(time.time())
                cv2.imwrite(f"kinect_rgb_{ts}.png", color_bgr)
                cv2.imwrite(f"kinect_depth_{ts}.png", depth_color)
                # 同时保存原始深度（便于后续分析）
                np.save(f"kinect_depth_raw_{ts}.npy", capture.depth)
                print(f"[INFO] 已保存截图: kinect_rgb_{ts}.png / kinect_depth_{ts}.png / kinect_depth_raw_{ts}.npy")
    finally:
        k4a.stop()
        cv2.destroyAllWindows()
        print("[INFO] Kinect 已关闭")


if __name__ == "__main__":
    main()
