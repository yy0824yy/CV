"""
基于深度图与人体关键点的距离测量。

提供功能：
    - 多关键点距离查询（鼻尖、左/右肩、左/右手腕、髋部中心）
    - 距离平滑（EMA，抑制单帧深度噪声）
    - 距离阈值告警（过近 / 过远）
    - 深度有效率统计（有效像素比例）
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

from core.depth_utils import get_distance_mm
from core.pose_detector import (
    PoseLandmarks,
    NOSE,
    LEFT_SHOULDER, RIGHT_SHOULDER,
    LEFT_WRIST, RIGHT_WRIST,
    LEFT_HIP, RIGHT_HIP,
)
from core.smoothing import EMASmoother


# 我们关心的关键点 → 输出键名
TRACK_POINTS: Dict[str, int] = {
    "nose": NOSE,
    "left_shoulder": LEFT_SHOULDER,
    "right_shoulder": RIGHT_SHOULDER,
    "left_wrist": LEFT_WRIST,
    "right_wrist": RIGHT_WRIST,
}


@dataclass
class DistanceReport:
    distances_m: Dict[str, float] = field(default_factory=dict)   # 各关键点 → 米；缺失则不在 dict 中
    body_distance_m: Optional[float] = None                       # 人体代表距离（鼻尖优先，否则肩中点）
    depth_valid_ratio: float = 0.0                                # 深度图有效像素比例 [0,1]
    too_close: bool = False                                       # 距离过近（< near_thr）
    too_far: bool = False                                         # 距离过远（> far_thr）


class DistanceEstimator:
    """多关键点距离测量器。"""

    def __init__(
        self,
        near_thr_m: float = 0.6,
        far_thr_m: float = 3.0,
        ema_alpha: float = 0.4,
        vis_thr: float = 0.5,
    ):
        self.near_thr_m = near_thr_m
        self.far_thr_m = far_thr_m
        self.vis_thr = vis_thr
        # 每个跟踪点一个 EMA
        self._smoothers: Dict[str, EMASmoother] = {
            k: EMASmoother(alpha=ema_alpha) for k in TRACK_POINTS.keys()
        }
        self._body_smoother = EMASmoother(alpha=ema_alpha)

    def reset(self):
        for s in self._smoothers.values():
            s.reset()
        self._body_smoother.reset()

    def estimate(
        self,
        depth_mm: Optional[np.ndarray],
        pose: Optional[PoseLandmarks],
    ) -> DistanceReport:
        """计算各点距离 + 深度有效率 + 警告标志。

        Args:
            depth_mm: 16-bit 深度图（毫米单位），shape=(H, W)
            pose: 姿态关键点，可为 None
        Returns:
            DistanceReport
        """
        rep = DistanceReport()
        if depth_mm is None:
            return rep

        # 深度有效率（有效像素 / 总像素）
        rep.depth_valid_ratio = float((depth_mm > 0).sum()) / float(depth_mm.size)

        if pose is None:
            return rep

        h_d, w_d = depth_mm.shape[:2]
        for key, idx in TRACK_POINTS.items():
            if not pose.is_visible(idx, self.vis_thr):
                continue
            nx = int(pose.landmarks_norm[idx][0] * w_d)
            ny = int(pose.landmarks_norm[idx][1] * h_d)
            d_mm = get_distance_mm(depth_mm, nx, ny, window=11)
            if d_mm is None:
                continue
            d_m = d_mm / 1000.0
            d_m = self._smoothers[key].update(d_m)
            rep.distances_m[key] = d_m

        # 人体代表距离：优先鼻尖；否则两肩中点（合成一个虚拟点）
        if "nose" in rep.distances_m:
            body = rep.distances_m["nose"]
        elif "left_shoulder" in rep.distances_m and "right_shoulder" in rep.distances_m:
            body = (rep.distances_m["left_shoulder"] + rep.distances_m["right_shoulder"]) / 2.0
        elif rep.distances_m:
            body = float(np.median(list(rep.distances_m.values())))
        else:
            body = None

        if body is not None:
            body = self._body_smoother.update(body)
            rep.body_distance_m = body
            rep.too_close = body < self.near_thr_m
            rep.too_far = body > self.far_thr_m

        return rep
