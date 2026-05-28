"""
跌倒检测器（基于姿态时序特征）。

核心思路：
    使用 3 个互补特征联合判定，避免单一特征误判：
        ① 躯干倾角  : 鼻子-髋部连线相对垂直方向的夹角
        ② 头部高度落差 : 一段时间窗口内头部 y 坐标的最大下降幅度
        ③ 倒地持续时间 : 必须保持"已倒下"状态足够长才告警

状态判定流程：
    每帧 -> 计算 3 个特征 -> 判定本帧是否"已倒下" ->
    维护"累计倒地时长" -> 累计满阈值且已过冷却期 -> 触发 FallEvent

设计要点：
    - 关键点不可见 / 无人 -> 重置累计器（不允许"挂着"过去的状态）
    - 冷却期：触发一次告警后 N 秒内不再触发，避免连续刷屏
    - 输出归一化坐标系下的特征值，便于上层 LLM 描述
"""
from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, Optional, Tuple

import numpy as np

from core.pose_detector import (
    PoseLandmarks,
    NOSE,
    LEFT_SHOULDER, RIGHT_SHOULDER,
    LEFT_HIP, RIGHT_HIP,
)


# ============================================================
# 数据结构
# ============================================================
class FallState(str, Enum):
    NORMAL = "normal"
    SUSPECT = "suspect"        # 当前帧符合"倒下特征"但持续时间不足
    FALLEN = "fallen"          # 已确认倒地
    COOLDOWN = "cooldown"      # 已告警，冷却中（避免重复告警）


@dataclass
class FallEvent:
    """一次跌倒告警事件。"""
    timestamp: float                # 触发时间（time.time()）
    confirmed_duration_sec: float   # 累计倒地确认时长
    torso_angle_deg: float          # 触发时刻躯干倾角
    head_drop: float                # 触发时刻头部下落幅度（0-1）
    person_distance_m: Optional[float] = None
    extra: dict = field(default_factory=dict)


# ============================================================
# 工具函数
# ============================================================
def _midpoint(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a + b) * 0.5


def _torso_angle_deg(shoulder_mid: np.ndarray, hip_mid: np.ndarray) -> float:
    """
    返回躯干向量（hip→shoulder）相对垂直方向（向上）的夹角（度）。

    说明：
        - 图像 y 轴向下，所以"向上"在归一化坐标里是 (0, -1)
        - 站立时躯干方向 ≈ (0, -1)，与垂直方向夹角 ≈ 0°
        - 平躺时躯干方向 ≈ (1, 0)，与垂直方向夹角 ≈ 90°
    """
    vec = shoulder_mid - hip_mid    # (dx, dy)，dy 为负代表向上
    norm = float(np.linalg.norm(vec[:2]))
    if norm < 1e-6:
        return 0.0
    # 垂直向上参考向量是 (0, -1)
    cos_theta = -vec[1] / norm
    cos_theta = max(-1.0, min(1.0, cos_theta))
    return math.degrees(math.acos(cos_theta))


# ============================================================
# 跌倒检测器
# ============================================================
class FallDetector:
    """
    每帧调用 update()，返回 None 或 FallEvent（仅在新触发时返回）。
    """

    def __init__(
        self,
        torso_tilt_threshold_deg: float = 55.0,
        head_drop_threshold: float = 0.20,
        confirm_seconds: float = 0.8,
        cooldown_seconds: float = 20.0,
        min_visibility: float = 0.4,
        history_seconds: float = 1.0,
    ):
        self.torso_tilt_threshold_deg = float(torso_tilt_threshold_deg)
        self.head_drop_threshold = float(head_drop_threshold)
        self.confirm_seconds = float(confirm_seconds)
        self.cooldown_seconds = float(cooldown_seconds)
        self.min_visibility = float(min_visibility)
        self.history_seconds = float(history_seconds)

        # 历史：(timestamp, nose_y, torso_angle_deg)
        self._history: Deque[Tuple[float, float, float]] = deque()
        self._fallen_since: Optional[float] = None
        self._last_alert_at: Optional[float] = None
        self._state: FallState = FallState.NORMAL
        self._last_features: dict = {}

    # ---------------- 状态读取 ----------------
    @property
    def state(self) -> FallState:
        return self._state

    @property
    def last_features(self) -> dict:
        """供 UI 显示当前指标用。"""
        return dict(self._last_features)

    def reset(self) -> None:
        self._history.clear()
        self._fallen_since = None
        self._last_alert_at = None
        self._state = FallState.NORMAL
        self._last_features = {}

    # ---------------- 主入口 ----------------
    def update(
        self,
        pose: Optional[PoseLandmarks],
        person_distance_m: Optional[float] = None,
    ) -> Optional[FallEvent]:
        now = time.time()

        # ---- 关键点缺失：清空累计但保留冷却 ----
        if pose is None or not self._key_points_visible(pose):
            self._fallen_since = None
            self._last_features = {"reason": "no_pose_or_low_visibility"}
            self._state = (
                FallState.COOLDOWN
                if self._in_cooldown(now) else FallState.NORMAL
            )
            return None

        lm = pose.landmarks_norm
        ls, rs = lm[LEFT_SHOULDER], lm[RIGHT_SHOULDER]
        lh, rh = lm[LEFT_HIP], lm[RIGHT_HIP]
        nose = lm[NOSE]
        shoulder_mid = _midpoint(ls, rs)
        hip_mid = _midpoint(lh, rh)

        # ---- 特征 1：躯干倾角 ----
        torso_angle = _torso_angle_deg(shoulder_mid, hip_mid)

        # ---- 特征 2：头部下落幅度（窗口最大值 vs 当前） ----
        nose_y = float(nose[1])
        self._history.append((now, nose_y, torso_angle))
        # 修剪 history 到窗口内
        cutoff = now - self.history_seconds
        while self._history and self._history[0][0] < cutoff:
            self._history.popleft()
        head_drop = 0.0
        if len(self._history) >= 2:
            min_y_in_window = min(h[1] for h in self._history)
            head_drop = nose_y - min_y_in_window  # 当前更大 = 下降

        # ---- 当前帧是否"已倒下" ----
        is_fallen_now = (
            torso_angle > self.torso_tilt_threshold_deg
            or head_drop > self.head_drop_threshold
        )

        # ---- 维护累计倒下时间 ----
        confirmed_duration = 0.0
        if is_fallen_now:
            if self._fallen_since is None:
                self._fallen_since = now
            confirmed_duration = now - self._fallen_since
        else:
            self._fallen_since = None

        # ---- 状态机 ----
        triggered_event: Optional[FallEvent] = None
        if self._in_cooldown(now):
            self._state = FallState.COOLDOWN
        elif is_fallen_now:
            if confirmed_duration >= self.confirm_seconds:
                self._state = FallState.FALLEN
                # 仅当从非告警状态过渡到 FALLEN 时触发一次
                triggered_event = FallEvent(
                    timestamp=now,
                    confirmed_duration_sec=confirmed_duration,
                    torso_angle_deg=torso_angle,
                    head_drop=head_drop,
                    person_distance_m=person_distance_m,
                )
                self._last_alert_at = now
            else:
                self._state = FallState.SUSPECT
        else:
            self._state = FallState.NORMAL

        # ---- 输出特征字典（UI / 调试用） ----
        self._last_features = {
            "torso_angle_deg": round(torso_angle, 1),
            "head_drop": round(head_drop, 3),
            "confirmed_duration_sec": round(confirmed_duration, 2),
            "is_fallen_now": is_fallen_now,
            "state": self._state.value,
        }

        return triggered_event

    # ---------------- 内部 ----------------
    def _key_points_visible(self, pose: PoseLandmarks) -> bool:
        for idx in (NOSE, LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP):
            if pose.visibility[idx] < self.min_visibility:
                return False
        return True

    def _in_cooldown(self, now: float) -> bool:
        if self._last_alert_at is None:
            return False
        return (now - self._last_alert_at) < self.cooldown_seconds
