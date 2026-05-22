"""
带标签的实验数据采集器（命令行交互式）。

用法：
    python experiments/collect_labeled.py
        --mode gesture                # 采集手势：依次提示 7 种手势
        --mode action                 # 采集动作：依次提示动作
        --mode distance               # 距离实验：在用户指定的距离下持续采集
        --frames 80                   # 每个标签采集多少帧（默认 80，约 5~6 秒）
        --prepare 3                   # 每个标签前的倒计时秒数（默认 3）

输出：
    data/outputs/labeled/<mode>_<timestamp>.csv
    列 = 原 17 列 CSV + 末尾两列 gt_label / gt_distance_m

设计：
    - 复用现有 VideoThread + DataLogger，避免重写采集逻辑
    - 用 Qt 事件循环跑后台采集，但本身是命令行（QApplication 也能跑 headless 计算）
    - 实际实现采用最小化方案：直接复用 core 各模块在本进程同步采集，更简单可控
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.camera import create_camera
from core.hand_detector import HandDetector
from core.pose_detector import PoseDetector
from core.stable_recognizers import StableGestureRecognizer, StableActionRecognizer
from core.geometry import compute_body_angles
from core.smoothing import LandmarkSmoother
from core.distance_estimator import DistanceEstimator
from data.logger import CSV_HEADERS
from config import CAMERA, DISPLAY


GESTURE_LABELS = [
    "Number_1", "Number_2", "Number_3",
    "Open_Palm", "Fist", "OK", "Thumbs_Up",
]

# 带标签 CSV 额外追加的列：hand_side + 21*3=63 个 landmark 值
# landmark 列名为 hx_0..hx_20, hy_0..hy_20, hz_0..hz_20
LANDMARK_COLS = (
    [f"hx_{i}" for i in range(21)]
    + [f"hy_{i}" for i in range(21)]
    + [f"hz_{i}" for i in range(21)]
)
EXTRA_COLS = ["gt_label", "gt_distance_m", "hand_side"] + LANDMARK_COLS
ACTION_LABELS = [
    "Standing", "Raise_Left_Hand", "Raise_Right_Hand", "Raise_Both_Hands",
    "Lean_Left", "Lean_Right", "Squat", "Bend_Over",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["gesture", "action", "distance"], default="gesture")
    p.add_argument("--frames", type=int, default=80, help="每个标签采集多少帧")
    p.add_argument("--prepare", type=int, default=3, help="每个标签开始前的倒计时秒数")
    p.add_argument("--source", default=CAMERA.source)
    p.add_argument("--distances", default="0.8,1.2,1.8,2.5,3.0",
                   help="distance 模式下要采集的距离档（米，逗号分隔）")
    return p.parse_args()


def make_csv(path: str):
    f = open(path, "w", newline="", encoding="utf-8")
    w = csv.writer(f)
    w.writerow(CSV_HEADERS + EXTRA_COLS)
    return f, w


def write_row(writer, processed_dict: dict, gt_label: str, gt_distance: float,
              hand_side: str = "", hand_lm: np.ndarray = None):
    """
    多写三块：原始 17 列 + (gt_label, gt_distance, hand_side) + 63 个 landmark 列。
    hand_lm: shape=(21, 3) 的 numpy 数组（归一化坐标）或 None。
    """
    a = processed_dict.get("angles", {}) or {}
    g_left = processed_dict.get("left_gesture", "")
    g_left_s = processed_dict.get("left_score", 0.0)
    g_right = processed_dict.get("right_gesture", "")
    g_right_s = processed_dict.get("right_score", 0.0)
    row = [
        f"{processed_dict.get('timestamp', 0):.3f}",
        datetime.now().isoformat(timespec="milliseconds"),
        f"{processed_dict.get('fps', 0):.2f}",
        g_left, f"{g_left_s:.3f}",
        g_right, f"{g_right_s:.3f}",
        processed_dict.get("action", ""),
        ("" if processed_dict.get("distance") is None
         else f"{processed_dict['distance']:.3f}"),
        f"{a.get('left_elbow', 0):.2f}",
        f"{a.get('right_elbow', 0):.2f}",
        f"{a.get('left_knee', 0):.2f}",
        f"{a.get('right_knee', 0):.2f}",
        f"{a.get('left_shoulder', 0):.2f}",
        f"{a.get('right_shoulder', 0):.2f}",
        f"{a.get('torso_tilt', 0):.2f}",
        f"{a.get('shoulder_tilt', 0):.2f}",
        gt_label,
        ("" if gt_distance is None else f"{gt_distance:.3f}"),
        hand_side,
    ]
    # 21*3 个 landmark：依次写 x[0..20], y[0..20], z[0..20]
    if hand_lm is not None and hand_lm.shape == (21, 3):
        for axis in range(3):
            for i in range(21):
                row.append(f"{hand_lm[i, axis]:.5f}")
    else:
        row.extend([""] * 63)
    writer.writerow(row)


def collect_one_label(
    cam, hand_det, pose_det, gesture_rec, action_rec,
    lm_smoother, dist_est,
    gt_label: str,
    gt_distance: float,
    frames: int,
    prepare_sec: int,
    csv_writer,
):
    print(f"\n>>> 即将采集标签 [{gt_label}]"
          + (f"  目标距离 {gt_distance:.2f}m" if gt_distance else "")
          + f"。{prepare_sec} 秒后开始 ...")
    # 倒计时阶段：仍要刷新画面，让用户摆好姿势
    t_end = time.time() + prepare_sec
    while time.time() < t_end:
        frame = cam.read()
        if frame is None:
            continue
        color = frame.color
        if DISPLAY.flip_horizontal:
            color = cv2.flip(color, 1)
        secs_left = max(0.0, t_end - time.time())
        cv2.putText(color, f"PREPARE: {gt_label}",
                    (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
        if gt_distance:
            cv2.putText(color, f"Target: {gt_distance:.2f}m",
                        (40, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(color, f"{secs_left:.1f}s",
                    (40, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        cv2.imshow("Collect", color)
        cv2.waitKey(1)

    print(f"    开始采集 {frames} 帧 ...")
    collected = 0
    while collected < frames:
        frame = cam.read()
        if frame is None:
            continue
        color = frame.color
        if DISPLAY.flip_horizontal:
            color = cv2.flip(color, 1)
        raw_depth = cv2.flip(frame.depth, 1) if (
            DISPLAY.flip_horizontal and frame.depth is not None
        ) else frame.depth

        hands = hand_det.detect(color)
        gestures = gesture_rec.recognize_all(hands)
        pose = pose_det.detect(color)
        if pose is not None:
            smoothed = lm_smoother.update(pose.landmarks_norm)
            pose.landmarks_norm = smoothed
            h, w = color.shape[:2]
            pose.landmarks_px[:, 0] = np.clip(smoothed[:, 0] * w, 0, w - 1)
            pose.landmarks_px[:, 1] = np.clip(smoothed[:, 1] * h, 0, h - 1)
        action = action_rec.recognize(pose)
        angles = compute_body_angles(pose.landmarks_norm) if pose is not None else {}
        dist_rep = dist_est.estimate(raw_depth, pose)

        left_g = ""
        left_s = 0.0
        right_g = ""
        right_s = 0.0
        for h_o, g in zip(hands, gestures):
            if h_o.handedness == "Left":
                left_g, left_s = g.name, g.score
            elif h_o.handedness == "Right":
                right_g, right_s = g.name, g.score

        # 选一只手作为该帧的训练样本：优先 右手（大多数人习惯手），
        # 没有右手则取第一个检测到的
        chosen_hand = None
        chosen_side = ""
        for h_o in hands:
            if h_o.handedness == "Right":
                chosen_hand = h_o
                chosen_side = "Right"
                break
        if chosen_hand is None and hands:
            chosen_hand = hands[0]
            chosen_side = chosen_hand.handedness

        write_row(
            csv_writer,
            {
                "timestamp": time.time(),
                "fps": 0.0,
                "left_gesture": left_g, "left_score": left_s,
                "right_gesture": right_g, "right_score": right_s,
                "action": action.primary if action else "",
                "distance": dist_rep.body_distance_m,
                "angles": angles,
            },
            gt_label, gt_distance,
            hand_side=chosen_side,
            hand_lm=chosen_hand.landmarks_norm if chosen_hand else None,
        )

        # 在画面上显示进度
        pose_det.draw(color, pose)
        hand_det.draw(color, hands)
        cv2.putText(color, f"COLLECTING: {gt_label}",
                    (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
        cv2.putText(color, f"{collected+1}/{frames}",
                    (40, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Collect", color)
        cv2.waitKey(1)
        collected += 1

    print(f"    [{gt_label}] 采集完成（{frames} 帧）")


def main():
    args = parse_args()

    # 选择标签集
    if args.mode == "gesture":
        labels = [(g, None) for g in GESTURE_LABELS]
    elif args.mode == "action":
        labels = [(a, None) for a in ACTION_LABELS]
    else:   # distance
        dist_list = [float(x) for x in args.distances.split(",")]
        labels = [("Standing", d) for d in dist_list]

    # 输出 CSV 路径
    out_dir = os.path.join("data", "outputs", "labeled")
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"{args.mode}_{ts}.csv")
    f_csv, w_csv = make_csv(out_path)
    print(f"[INFO] 输出 CSV: {out_path}")

    # 摄像头
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

    hand_det = HandDetector(max_num_hands=2, already_flipped=False)
    pose_det = PoseDetector(model_complexity=1)
    gesture_rec = StableGestureRecognizer(window=7)
    action_rec = StableActionRecognizer(window=9)
    lm_smoother = LandmarkSmoother(alpha=0.7)
    dist_est = DistanceEstimator()

    try:
        cam.open()
        for (label, dist) in labels:
            collect_one_label(
                cam, hand_det, pose_det, gesture_rec, action_rec,
                lm_smoother, dist_est,
                gt_label=label, gt_distance=dist,
                frames=args.frames, prepare_sec=args.prepare,
                csv_writer=w_csv,
            )
        print(f"\n[INFO] 全部完成。CSV 已保存: {out_path}")
    finally:
        f_csv.close()
        try:
            cam.close()
        except Exception:
            pass
        hand_det.close()
        pose_det.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
