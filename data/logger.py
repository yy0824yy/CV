"""
数据记录模块。

封装四类数据记录：
    1) 截图（彩色 + 深度伪彩 + 原始深度 npy + 关键点 JSON）
    2) 视频录制（mp4）
    3) CSV 实时日志（每帧追加一行）
    4) 关键点 JSON 快照

输出根目录：data/outputs/
    snapshots/    截图与对应 JSON
    videos/       录制视频
    logs/         CSV 日志
"""
from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime
from typing import List, Optional

import cv2
import numpy as np


# CSV 列定义（顺序固定，便于后续 pandas 分析）
CSV_HEADERS = [
    "timestamp", "datetime", "fps",
    "left_gesture", "left_score",
    "right_gesture", "right_score",
    "action",
    "person_distance_m",
    "left_elbow", "right_elbow",
    "left_knee", "right_knee",
    "left_shoulder", "right_shoulder",
    "torso_tilt", "shoulder_tilt",
]


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _resolve_root(root: str) -> str:
    if os.path.isabs(root):
        return root
    return os.path.join(PROJECT_ROOT, root)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _ts_str() -> str:
    """文件名安全的时间戳：20260513_201505_123."""
    now = datetime.now()
    return now.strftime("%Y%m%d_%H%M%S_") + f"{int(now.microsecond/1000):03d}"


class DataLogger:
    """统一数据记录器。

    用法：
        logger = DataLogger(root="data/outputs")
        logger.save_snapshot(processed_frame)              # 截图 + JSON
        logger.start_recording(processed_frame)            # 开始录视频
        logger.append_video_frame(processed_frame.bgr)     # 每帧追加
        logger.stop_recording()                            # 结束
        logger.start_csv_log()                             # 开始 CSV
        logger.append_csv(processed_frame)                 # 每帧追加
        logger.stop_csv_log()
    """

    def __init__(self, root: str = "data/outputs"):
        self.root = _resolve_root(root)
        self.snap_dir = os.path.join(self.root, "snapshots")
        self.video_dir = os.path.join(self.root, "videos")
        self.log_dir = os.path.join(self.root, "logs")
        _ensure_dir(self.snap_dir)
        _ensure_dir(self.video_dir)
        _ensure_dir(self.log_dir)

        # 视频录制状态
        self._writer: Optional[cv2.VideoWriter] = None
        self._video_path: Optional[str] = None

        # CSV 日志状态
        self._csv_file = None
        self._csv_writer = None
        self._csv_path: Optional[str] = None

    # ------------------------------------------------------------
    # 1) 截图
    # ------------------------------------------------------------
    def save_snapshot(self, processed) -> dict:
        """保存当前帧的截图与关键点 JSON。

        Args:
            processed: ProcessedFrame 实例（来自 ui.video_thread）

        Returns:
            dict: 包含所有保存路径，便于上层显示
        """
        ts = _ts_str()
        paths = {}
        # 已叠加可视化的彩色图
        if processed.bgr is not None:
            p = os.path.join(self.snap_dir, f"vis_{ts}.png")
            cv2.imwrite(p, processed.bgr)
            paths["visualized"] = p
        # 原始彩色（无叠加）
        if processed.raw_bgr is not None:
            p = os.path.join(self.snap_dir, f"color_{ts}.png")
            cv2.imwrite(p, processed.raw_bgr)
            paths["color"] = p
        # 深度伪彩
        if processed.depth_color is not None:
            p = os.path.join(self.snap_dir, f"depth_{ts}.png")
            cv2.imwrite(p, processed.depth_color)
            paths["depth_color"] = p
        # 原始深度（用于离线分析）
        if processed.raw_depth is not None:
            p = os.path.join(self.snap_dir, f"depth_raw_{ts}.npy")
            np.save(p, processed.raw_depth)
            paths["depth_raw"] = p

        # 关键点 + 识别结果 JSON
        json_data = {
            "timestamp": time.time(),
            "datetime": datetime.now().isoformat(timespec="milliseconds"),
            "fps": float(processed.fps),
            "person_distance_m": (
                float(processed.person_distance_m)
                if processed.person_distance_m is not None else None
            ),
            "action": (
                {
                    "primary": processed.action.primary,
                    "labels": list(processed.action.labels),
                    "valid": bool(processed.action.valid),
                    "features": dict(processed.action.features),
                } if processed.action is not None else None
            ),
            "angles": {k: float(v) for k, v in (processed.angles or {}).items()},
            "hands": [
                {
                    "handedness": h.handedness,
                    "score": float(h.score),
                    "landmarks_norm": h.landmarks_norm.tolist(),
                }
                for h in processed.hands
            ],
            "gestures": [
                {
                    "name": g.name,
                    "score": float(g.score),
                    "finger_states": list(g.finger_states),
                    "extra": {k: float(v) for k, v in g.extra.items()},
                }
                for g in processed.gestures
            ],
            "pose": (
                {
                    "landmarks_norm": processed.pose.landmarks_norm.tolist(),
                    "visibility": processed.pose.visibility.tolist(),
                } if processed.pose is not None else None
            ),
        }
        p = os.path.join(self.snap_dir, f"data_{ts}.json")
        with open(p, "w", encoding="utf-8") as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        paths["json"] = p
        return paths

    # ------------------------------------------------------------
    # 2) 视频录制
    # ------------------------------------------------------------
    def is_recording(self) -> bool:
        return self._writer is not None

    def start_recording(self, processed, fps: float = 25.0) -> str:
        """开始录制视频，使用 processed.bgr 的尺寸初始化。"""
        if self._writer is not None:
            return self._video_path  # 已在录制
        if processed is None or processed.bgr is None:
            raise RuntimeError("无可用画面，无法开始录制")
        h, w = processed.bgr.shape[:2]
        ts = _ts_str()
        path = os.path.join(self.video_dir, f"record_{ts}.mp4")
        # mp4v 编码器在 Windows 上无需额外依赖
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(path, fourcc, float(fps), (w, h))
        if not self._writer.isOpened():
            self._writer = None
            raise RuntimeError("VideoWriter 无法打开，请检查 OpenCV 编码器")
        self._video_path = path
        return path

    def append_video_frame(self, bgr: np.ndarray) -> None:
        if self._writer is None or bgr is None:
            return
        self._writer.write(bgr)

    def stop_recording(self) -> Optional[str]:
        """结束录制并返回视频路径。"""
        if self._writer is None:
            return None
        self._writer.release()
        path = self._video_path
        self._writer = None
        self._video_path = None
        return path

    # ------------------------------------------------------------
    # 3) CSV 实时日志
    # ------------------------------------------------------------
    def is_csv_logging(self) -> bool:
        return self._csv_writer is not None

    def start_csv_log(self) -> str:
        if self._csv_writer is not None:
            return self._csv_path
        ts = _ts_str()
        path = os.path.join(self.log_dir, f"log_{ts}.csv")
        f = open(path, "w", newline="", encoding="utf-8")
        w = csv.writer(f)
        w.writerow(CSV_HEADERS)
        self._csv_file = f
        self._csv_writer = w
        self._csv_path = path
        return path

    def append_csv(self, processed) -> None:
        if self._csv_writer is None or processed is None:
            return
        # 取左右手手势（按 handedness 匹配）
        left_g, left_s, right_g, right_s = "", 0.0, "", 0.0
        for h, g in zip(processed.hands, processed.gestures):
            if h.handedness == "Left":
                left_g, left_s = g.name, g.score
            elif h.handedness == "Right":
                right_g, right_s = g.name, g.score
        a = processed.angles or {}
        action_name = processed.action.primary if processed.action is not None else ""
        row = [
            f"{processed.timestamp:.3f}",
            datetime.now().isoformat(timespec="milliseconds"),
            f"{processed.fps:.2f}",
            left_g, f"{left_s:.3f}",
            right_g, f"{right_s:.3f}",
            action_name,
            "" if processed.person_distance_m is None
            else f"{processed.person_distance_m:.3f}",
            f"{a.get('left_elbow', 0):.2f}",
            f"{a.get('right_elbow', 0):.2f}",
            f"{a.get('left_knee', 0):.2f}",
            f"{a.get('right_knee', 0):.2f}",
            f"{a.get('left_shoulder', 0):.2f}",
            f"{a.get('right_shoulder', 0):.2f}",
            f"{a.get('torso_tilt', 0):.2f}",
            f"{a.get('shoulder_tilt', 0):.2f}",
        ]
        self._csv_writer.writerow(row)

    def stop_csv_log(self) -> Optional[str]:
        if self._csv_writer is None:
            return None
        try:
            self._csv_file.flush()
            self._csv_file.close()
        except Exception:
            pass
        path = self._csv_path
        self._csv_writer = None
        self._csv_file = None
        self._csv_path = None
        return path

    # ------------------------------------------------------------
    # 关闭：保证退出时不丢资源
    # ------------------------------------------------------------
    def close_all(self):
        self.stop_recording()
        self.stop_csv_log()
