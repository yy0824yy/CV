"""
MediaPipe Pose 与 Azure Kinect Body Tracking SDK 对比实验。

同时在同一 Kinect 设备上跑两种姿态估计方法，量化对比：
    1) 检测率（detection rate）—— 每帧是否成功检测出至少一个人
    2) 处理速度（latency）—— 每帧推理耗时
    3) 关键点稳定性 —— 7 个共同关节的位置标准差

报告里这是"对比实验"那一章的核心。

用法：
    python experiments/compare_pose.py --duration 30
        --duration 30        测量时长（秒），默认 30
        --output FILENAME    输出 CSV 与图表前缀，默认 compare_<timestamp>
        --no-display         不显示实时画面（仅采集数据）
        --hold-still         提示用户保持静止（更适合测稳定性）

输出：
    data/outputs/figures/compare_<ts>.png          对比图
    data/outputs/labeled/compare_<ts>.csv          每帧逐项数据

测量说明：
    MediaPipe 输出是图像 2D 归一化坐标，本脚本转为像素单位
    Body Tracking 输出是世界 3D 坐标（毫米），本脚本投影回彩色相机 2D 像素
    这样两个 backend 的稳定性指标都是 "像素 std"，可直接对比。
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from datetime import datetime

import cv2
import numpy as np

# 项目路径
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# Body Tracking SDK
import pykinect_azure as pykinect

# MediaPipe（仅 Pose，不用我们的 PoseDetector 类，确保对比公平）
import mediapipe as mp


# ============================================================
# 共同关节定义
# ============================================================
# MediaPipe 33 关键点的索引（标准定义）
MP_JOINTS = {
    "nose": 0,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
}

# K4ABT 32 关节的索引（标准定义）
K4ABT_JOINTS = {
    "nose": 27,
    "left_shoulder": 5,
    "right_shoulder": 12,
    "left_elbow": 6,
    "right_elbow": 13,
    "left_wrist": 7,
    "right_wrist": 14,
}

COMMON_JOINTS = list(MP_JOINTS.keys())  # 7 个共同关节


# ============================================================
# 工具：K4ABT body 提取 2D 像素坐标
# ============================================================
def _extract_body_pixels(body, calibration, image_shape):
    """从 K4ABT body 提取 32 关节的 (x_px, y_px, confidence)。

    优先用 body.numpy() 提供的 2D 像素坐标，失败则用 3D 投影。
    返回 ndarray (32, 3) 或 None。
    """
    try:
        arr = body.numpy()  # shape: (32, ?)
        # 不同版本 pykinect_azure 的列顺序略有差异。
        # 常见格式：[x_3d, y_3d, z_3d, conf, w?, x?, y?, z?] 或
        # [x_3d, y_3d, z_3d, conf, x_2d, y_2d, conf_2d] (k4abt_joint2D_t)
        if arr.ndim != 2 or arr.shape[0] < 32:
            return None
    except Exception:
        return None

    # 尝试找到 2D 像素列
    # 如果有 7 列，通常后 3 列是 2D
    if arr.shape[1] >= 7:
        return arr[:32, 4:7].astype(np.float32)  # (32, 3) = [x_2d, y_2d, conf]

    # 退化：从 3D 自行投影
    h, w = image_shape[:2]
    out = np.zeros((32, 3), dtype=np.float32)
    for j in range(min(32, arr.shape[0])):
        x_3d, y_3d, z_3d = arr[j, 0], arr[j, 1], arr[j, 2]
        if z_3d <= 0:
            continue
        try:
            # 用 calibration 投影 3D→2D
            pos2d = calibration.convert_3d_to_2d(
                (x_3d, y_3d, z_3d),
                pykinect.K4A_CALIBRATION_TYPE_DEPTH,
                pykinect.K4A_CALIBRATION_TYPE_COLOR,
            )
            if pos2d is not None:
                out[j, 0] = pos2d[0]
                out[j, 1] = pos2d[1]
                out[j, 2] = arr[j, 3] if arr.shape[1] > 3 else 1.0
        except Exception:
            pass
    return out


# ============================================================
# 主流程
# ============================================================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--duration", type=float, default=30.0)
    p.add_argument("--output", default=None,
                   help="输出文件前缀（不含扩展名）")
    p.add_argument("--no-display", action="store_true")
    p.add_argument("--hold-still", action="store_true",
                   help="开始前 3 秒提示保持静止（更适合测稳定性）")
    return p.parse_args()


def main():
    args = parse_args()

    # ---------- 输出路径 ----------
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = args.output or f"compare_{ts}"
    fig_dir = os.path.join(ROOT, "data", "outputs", "figures")
    csv_dir = os.path.join(ROOT, "data", "outputs", "labeled")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)
    fig_path = os.path.join(fig_dir, f"{prefix}.png")
    csv_path = os.path.join(csv_dir, f"{prefix}.csv")
    print(f"[INFO] 输出: {fig_path}\n        {csv_path}")

    # ---------- 启动 Kinect + Body Tracking ----------
    print("[INFO] 初始化 Body Tracking SDK ...")
    pykinect.initialize_libraries(track_body=True)
    config = pykinect.default_configuration
    config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_720P
    config.depth_mode = pykinect.K4A_DEPTH_MODE_NFOV_UNBINNED
    config.camera_fps = pykinect.K4A_FRAMES_PER_SECOND_15
    config.synchronized_images_only = True

    print("[INFO] 启动设备 ...")
    device = pykinect.start_device(config=config)
    calibration = device.get_calibration(
        pykinect.K4A_DEPTH_MODE_NFOV_UNBINNED,
        pykinect.K4A_COLOR_RESOLUTION_720P,
    )

    print("[INFO] 启动 Body Tracker（首次约 5~15 秒）...")
    body_tracker = pykinect.start_body_tracker()

    # ---------- MediaPipe Pose ----------
    print("[INFO] 加载 MediaPipe Pose ...")
    mp_pose = mp.solutions.pose.Pose(
        static_image_mode=False, model_complexity=1,
        enable_segmentation=False, min_detection_confidence=0.5,
    )

    # ---------- 倒计时 ----------
    if args.hold_still:
        print("[INFO] 3 秒后开始，请站到镜头前并保持静止...")
        for i in range(3, 0, -1):
            print(f"  {i} ...")
            time.sleep(1)

    print(f"[INFO] 采集 {args.duration:.0f} 秒。q 提前结束。")

    # ---------- 数据记录 ----------
    # 每帧一项
    records = []
    # 7 个关节的历史 2D 位置 [(x,y), ...] 各 backend 各一份
    mp_joint_hist = {j: [] for j in COMMON_JOINTS}
    bt_joint_hist = {j: [] for j in COMMON_JOINTS}

    t_start = time.time()
    frame_idx = 0

    try:
        while time.time() - t_start < args.duration:
            capture = device.update()
            body_frame = body_tracker.update()

            ret_color, color_image = capture.get_color_image()
            if not ret_color:
                continue
            if color_image.shape[-1] == 4:
                color_bgr = cv2.cvtColor(color_image, cv2.COLOR_BGRA2BGR)
            else:
                color_bgr = color_image

            h, w = color_bgr.shape[:2]

            # ---------- MediaPipe ----------
            t0 = time.perf_counter()
            rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
            mp_result = mp_pose.process(rgb)
            t_mp = (time.perf_counter() - t0) * 1000.0

            mp_detected = mp_result.pose_landmarks is not None
            mp_joints_px = {}
            if mp_detected:
                for name, idx in MP_JOINTS.items():
                    lm = mp_result.pose_landmarks.landmark[idx]
                    px = (lm.x * w, lm.y * h)
                    mp_joints_px[name] = px
                    mp_joint_hist[name].append(px)

            # ---------- Body Tracking ----------
            t0 = time.perf_counter()
            n_bodies = body_frame.get_num_bodies()
            t_bt = (time.perf_counter() - t0) * 1000.0
            # 注意：BT 的真正推理发生在 device.update / body_tracker.update 阶段，
            # 上面的 t_bt 只是查询时间。我们用 capture/tracker.update 两步合计作为代价。
            # 这里近似处理，详见 README。

            bt_detected = n_bodies > 0
            bt_joints_px = {}
            if bt_detected:
                body = body_frame.get_body(0)
                pix = _extract_body_pixels(body, calibration, color_bgr.shape)
                if pix is not None:
                    for name, idx in K4ABT_JOINTS.items():
                        x, y, conf = pix[idx]
                        if conf > 0 and 0 < x < w and 0 < y < h:
                            bt_joints_px[name] = (float(x), float(y))
                            bt_joint_hist[name].append((float(x), float(y)))

            # ---------- 记录 ----------
            records.append({
                "frame": frame_idx,
                "t_rel": time.time() - t_start,
                "mp_detected": int(mp_detected),
                "bt_detected": int(bt_detected),
                "mp_time_ms": t_mp,
                "bt_query_time_ms": t_bt,
                "n_bodies_bt": n_bodies,
            })

            # ---------- 实时显示（可选） ----------
            if not args.no_display:
                disp = color_bgr.copy()
                # 绘 MediaPipe 关键点（红色）
                for name, (x, y) in mp_joints_px.items():
                    cv2.circle(disp, (int(x), int(y)), 6, (0, 0, 255), -1)
                # 绘 Body Tracking 关键点（绿色）
                for name, (x, y) in bt_joints_px.items():
                    cv2.circle(disp, (int(x), int(y)), 6, (0, 255, 0), 2)

                elapsed = time.time() - t_start
                cv2.rectangle(disp, (0, 0), (disp.shape[1], 40),
                              (40, 40, 40), -1)
                cv2.putText(
                    disp,
                    f"[{elapsed:5.1f}s] MP(red)={'OK' if mp_detected else '--'} "
                    f"{t_mp:5.1f}ms  |  "
                    f"BT(green)={'OK' if bt_detected else '--'} n={n_bodies}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (0, 255, 255), 2,
                )

                if disp.shape[1] > 1280:
                    disp = cv2.resize(disp, (1280, 720))
                cv2.imshow("Compare: MediaPipe (red) vs Body Tracking (green)",
                           disp)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("[INFO] 用户中断")
                    break

            frame_idx += 1

    finally:
        try:
            device.close()
        except Exception:
            pass
        try:
            mp_pose.close()
        except Exception:
            pass
        cv2.destroyAllWindows()

    if not records:
        print("[ERROR] 没采到任何数据，退出")
        return

    # ============================================================
    # 统计 + 图表
    # ============================================================
    n_total = len(records)
    n_mp_det = sum(r["mp_detected"] for r in records)
    n_bt_det = sum(r["bt_detected"] for r in records)
    mp_time = np.array([r["mp_time_ms"] for r in records], dtype=np.float32)
    # BT 实际推理时间近似为帧间隔（限于 SDK 同步调用方式）
    # 这里直接报告 query 时间（极小），并在报告里说明
    bt_query = np.array([r["bt_query_time_ms"] for r in records], dtype=np.float32)

    # 关节稳定性：每个关节的 (x, y) 标准差，再取 xy 平均
    def joint_std(hist: dict) -> dict:
        out = {}
        for name, pts in hist.items():
            if len(pts) < 5:
                out[name] = float("nan")
                continue
            arr = np.array(pts, dtype=np.float32)
            std_x = float(arr[:, 0].std())
            std_y = float(arr[:, 1].std())
            out[name] = (std_x + std_y) / 2.0
        return out

    mp_std = joint_std(mp_joint_hist)
    bt_std = joint_std(bt_joint_hist)

    # ---------- 控制台总结 ----------
    print("\n" + "=" * 60)
    print(f"采集帧数: {n_total}")
    print(f"检测率:    MediaPipe {n_mp_det/n_total*100:.1f}%  |  "
          f"Body Tracking {n_bt_det/n_total*100:.1f}%")
    print(f"推理耗时:  MediaPipe {mp_time.mean():.1f}±{mp_time.std():.1f} ms")
    print(f"关键点平均像素标准差（越低越稳定）:")
    for j in COMMON_JOINTS:
        ms = mp_std.get(j, float("nan"))
        bs = bt_std.get(j, float("nan"))
        print(f"  {j:18s}  MP {ms:6.2f}px  |  BT {bs:6.2f}px")
    print("=" * 60)

    # ---------- 写 CSV ----------
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["frame", "t_rel", "mp_detected", "bt_detected",
                    "mp_time_ms", "bt_query_time_ms", "n_bodies_bt"])
        for r in records:
            w.writerow([
                r["frame"], f"{r['t_rel']:.3f}",
                r["mp_detected"], r["bt_detected"],
                f"{r['mp_time_ms']:.2f}", f"{r['bt_query_time_ms']:.2f}",
                r["n_bodies_bt"],
            ])
    print(f"[OUT] CSV: {csv_path}")

    # ---------- 画图 ----------
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # ① 检测率
    ax = axes[0, 0]
    bars = ax.bar(
        ["MediaPipe", "Body Tracking"],
        [n_mp_det / n_total * 100, n_bt_det / n_total * 100],
        color=["#EF5350", "#43A047"], edgecolor="#222",
    )
    for b, v in zip(bars, [n_mp_det / n_total * 100, n_bt_det / n_total * 100]):
        ax.text(b.get_x() + b.get_width()/2, v + 1,
                f"{v:.1f}%", ha="center", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 110)
    ax.set_ylabel("检测率 (%)")
    ax.set_title(f"人体检测率（N={n_total} 帧）")
    ax.grid(True, alpha=0.3, axis="y")

    # ② 推理耗时
    ax = axes[0, 1]
    mp_mean, mp_s = float(mp_time.mean()), float(mp_time.std())
    ax.bar(["MediaPipe"], [mp_mean], yerr=[mp_s],
           color="#EF5350", edgecolor="#222", capsize=8)
    ax.text(0, mp_mean + mp_s, f"{mp_mean:.1f}±{mp_s:.1f}ms",
            ha="center", fontsize=11, fontweight="bold")
    ax.set_ylabel("单帧推理时间 (ms)")
    ax.set_title("MediaPipe 处理速度\n(BT 同步在 device.update 内部，"
                 "未单独计时)")
    ax.grid(True, alpha=0.3, axis="y")

    # ③ 关键点稳定性（柱状对比）
    ax = axes[1, 0]
    x = np.arange(len(COMMON_JOINTS))
    width = 0.38
    mp_vals = [mp_std.get(j, 0) if not np.isnan(mp_std.get(j, np.nan))
               else 0 for j in COMMON_JOINTS]
    bt_vals = [bt_std.get(j, 0) if not np.isnan(bt_std.get(j, np.nan))
               else 0 for j in COMMON_JOINTS]
    ax.bar(x - width/2, mp_vals, width, label="MediaPipe",
           color="#EF5350", edgecolor="#222")
    ax.bar(x + width/2, bt_vals, width, label="Body Tracking",
           color="#43A047", edgecolor="#222")
    ax.set_xticks(x)
    ax.set_xticklabels(COMMON_JOINTS, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("位置标准差 (px)")
    ax.set_title("关键点稳定性对比（越低越稳定）")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # ④ 检测对齐：每帧两 backend 同时检出/单边检出
    ax = axes[1, 1]
    both = sum(1 for r in records if r["mp_detected"] and r["bt_detected"])
    only_mp = sum(1 for r in records if r["mp_detected"] and not r["bt_detected"])
    only_bt = sum(1 for r in records if not r["mp_detected"] and r["bt_detected"])
    neither = sum(1 for r in records if not r["mp_detected"] and not r["bt_detected"])
    sizes = [both, only_mp, only_bt, neither]
    labels = [f"两者都检出\n{both}", f"仅 MP\n{only_mp}",
              f"仅 BT\n{only_bt}", f"两者都失败\n{neither}"]
    colors_pie = ["#66BB6A", "#FFA726", "#42A5F5", "#BDBDBD"]
    nonzero = [(s, l, c) for s, l, c in zip(sizes, labels, colors_pie) if s > 0]
    if nonzero:
        ss, ll, cc = zip(*nonzero)
        ax.pie(ss, labels=ll, colors=cc, autopct="%1.0f%%",
               textprops={"fontsize": 10})
        ax.set_title("逐帧检测一致性")
    else:
        ax.axis("off")

    fig.suptitle(
        f"MediaPipe Pose vs Azure Kinect Body Tracking SDK 对比实验\n"
        f"采集时长 {args.duration:.0f}s | 总帧数 {n_total}",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"[OUT] 图表: {fig_path}")
    print("[完成]")


if __name__ == "__main__":
    main()
