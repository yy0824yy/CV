"""
最小验证脚本 1：用 OpenCV 打开默认摄像头（笔记本/USB），实时显示 RGB + FPS。
按 q 退出，按 s 截图。

用途：确认 OpenCV + 摄像头工作正常。这一步不依赖 Azure Kinect。
"""
import cv2
import time


def main():
    # 打开 0 号摄像头（笔记本默认或第一个 USB 摄像头）
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Windows 上加 CAP_DSHOW 启动更快
    if not cap.isOpened():
        print("[ERROR] 无法打开摄像头，请检查设备")
        return

    print("[INFO] OpenCV 摄像头已打开。按 q 退出，按 s 截图。")
    prev_t = time.time()
    fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] 读取帧失败")
            break

        # 计算 FPS（指数平滑，更稳）
        now = time.time()
        inst_fps = 1.0 / max(now - prev_t, 1e-6)
        fps = fps * 0.9 + inst_fps * 0.1 if fps > 0 else inst_fps
        prev_t = now

        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.imshow("OpenCV Camera Test (q=quit, s=save)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            fname = f"opencv_snapshot_{int(time.time())}.png"
            cv2.imwrite(fname, frame)
            print(f"[INFO] 已保存截图: {fname}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
