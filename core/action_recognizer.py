"""
基于 Pose 33 关键点的人体动作识别（规则方法）。

支持动作（多标签）：
    Raise_Left_Hand, Raise_Right_Hand, Raise_Both_Hands,
    Squat, Bend_Over, Lean_Left, Lean_Right,
    Standing（兜底）

设计要点：
    - 所有阈值基于"身体比例"而非像素距离，距离镜头远近不影响判定
    - 关键点不可见（visibility 太低）时拒绝判定，避免误识别
    - 返回多标签 + 一个 primary（优先级最高的展示用）
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from core.pose_detector import (
    PoseLandmarks,
    LEFT_SHOULDER, RIGHT_SHOULDER,
    LEFT_ELBOW, RIGHT_ELBOW,
    LEFT_WRIST, RIGHT_WRIST,
    LEFT_HIP, RIGHT_HIP,
    LEFT_KNEE, RIGHT_KNEE,
    LEFT_ANKLE, RIGHT_ANKLE,
    NOSE,
)


# 动作优先级（越靠前越优先作为 primary 展示）
ACTION_PRIORITY = [
    "Raise_Both_Hands",
    "Raise_Left_Hand",
    "Raise_Right_Hand",
    "Squat",
    "Bend_Over",
    "Lean_Left",
    "Lean_Right",
    "Standing",
]


@dataclass
class ActionResult:
    """动作识别结果。"""
    primary: str                           # 主动作（优先级最高的）
    labels: List[str] = field(default_factory=list)   # 同时满足的所有动作
    features: Dict[str, float] = field(default_factory=dict)  # 调试用特征
    valid: bool = True                     # 关键点是否足够可信


# ============================================================
# 几何工具
# ============================================================
def _midpoint(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a + b) * 0.5


def _angle_deg(v: np.ndarray, ref: np.ndarray) -> float:
    """两个 2D 向量的夹角（度），输入 (x, y)。"""
    v = v[:2]; ref = ref[:2]
    nv = np.linalg.norm(v) * np.linalg.norm(ref)
    if nv < 1e-6:
        return 0.0
    cos = float(np.dot(v, ref) / nv)
    cos = max(-1.0, min(1.0, cos))
    return math.degrees(math.acos(cos))


# ============================================================
# 动作识别主类
# ============================================================
class ActionRecognizer:
    """规则式动作识别器。"""

    def __init__(
        self,
        vis_thr: float = 0.35,
        hand_above_margin: float = 0.03,      # 手腕高于肩膀的最小归一化差值
        shoulder_tilt_deg: float = 12.0,      # 肩线倾斜超过该角度判定为左右倾
        torso_tilt_deg: float = 22.0,         # 躯干倾斜超过该角度判定为弯腰
        squat_knee_hip_ratio: float = 0.65,   # 膝-髋距 / 身高（肩-髋）小于该值判定下蹲
    ):
        self.vis_thr = vis_thr
        self.hand_above_margin = hand_above_margin
        self.shoulder_tilt_deg = shoulder_tilt_deg
        self.torso_tilt_deg = torso_tilt_deg
        self.squat_knee_hip_ratio = squat_knee_hip_ratio

    def _visible(self, pose: PoseLandmarks, indices: List[int]) -> bool:
        return all(pose.visibility[i] >= self.vis_thr for i in indices)

    def recognize(self, pose: Optional[PoseLandmarks]) -> ActionResult:
        if pose is None:
            return ActionResult(primary="No_Person", labels=[], valid=False)

        lm = pose.landmarks_norm   # 归一化 (x, y, z)，y 越小越靠上
        labels: List[str] = []
        features: Dict[str, float] = {}

        # ----- 必备关键点：肩 + 髋 -----
        upper_ok = self._visible(pose, [LEFT_SHOULDER, RIGHT_SHOULDER])
        hip_ok = self._visible(pose, [LEFT_HIP, RIGHT_HIP])

        if not upper_ok:
            # 连双肩都不可信，无法做任何判定
            return ActionResult(primary="Unknown", labels=[], valid=False)

        ls, rs = lm[LEFT_SHOULDER], lm[RIGHT_SHOULDER]
        shoulder_mid = _midpoint(ls, rs)

        # ===== 1) 举手 =====
        # 注意 MediaPipe 的 LEFT 是被拍者自己的左侧。
        # 但由于我们对画面做了镜像（自拍视角），用户看到的"左"
        # 就是被拍者的"右"。为了让 UI 上 "Left/Right" 与用户视角一致，
        # 这里把命名做一次对调：当 RIGHT_WRIST（被拍者右手）举起时，
        # 输出 "Raise_Left_Hand"（用户视角的"左手"）。
        if pose.visibility[LEFT_WRIST] >= self.vis_thr:
            lw = lm[LEFT_WRIST]
            left_above = (ls[1] - lw[1]) > self.hand_above_margin
            features["left_wrist_above"] = float(ls[1] - lw[1])
            if left_above:
                # 被拍者左手 = 用户视角右手
                labels.append("Raise_Right_Hand")

        if pose.visibility[RIGHT_WRIST] >= self.vis_thr:
            rw = lm[RIGHT_WRIST]
            right_above = (rs[1] - rw[1]) > self.hand_above_margin
            features["right_wrist_above"] = float(rs[1] - rw[1])
            if right_above:
                labels.append("Raise_Left_Hand")

        if "Raise_Left_Hand" in labels and "Raise_Right_Hand" in labels:
            # 合并为举双手
            labels = [x for x in labels if x not in
                      ("Raise_Left_Hand", "Raise_Right_Hand")]
            labels.append("Raise_Both_Hands")

        # ===== 2) 肩线倾斜：左右倾 =====
        # 计算两肩连线与水平方向夹角
        # 镜像视角：左肩在画面右侧，右肩在画面左侧
        # 用 dy / dx 算角度
        dx = rs[0] - ls[0]
        dy = rs[1] - ls[1]
        shoulder_angle = math.degrees(math.atan2(dy, dx))   # 水平时约 180° 或 0°
        # 转成相对水平的偏角（-90~90）
        shoulder_tilt = (shoulder_angle + 180) % 180
        if shoulder_tilt > 90:
            shoulder_tilt -= 180
        features["shoulder_tilt_deg"] = float(shoulder_tilt)

        if abs(shoulder_tilt) > self.shoulder_tilt_deg:
            # 判断哪边低（y 大）
            # 镜像视角下：left_shoulder（被拍者左肩，画面右侧）y 大 → 用户视角向"右"倾？
            # 简化：直接看 ls.y 与 rs.y 谁更大
            if ls[1] > rs[1]:
                # 被拍者左肩低 → 用户视角是身体向画面右侧倾
                labels.append("Lean_Right")
            else:
                labels.append("Lean_Left")

        # ===== 3) 弯腰：躯干前倾 =====
        if hip_ok:
            lh, rh = lm[LEFT_HIP], lm[RIGHT_HIP]
            hip_mid = _midpoint(lh, rh)
            # 躯干向量（髋指向肩，即从下到上）
            torso = shoulder_mid - hip_mid
            # 与竖直向上 (0, -1) 的夹角
            up = np.array([0.0, -1.0], dtype=np.float32)
            torso_tilt = _angle_deg(torso, up)
            features["torso_tilt_deg"] = float(torso_tilt)

            if torso_tilt > self.torso_tilt_deg:
                labels.append("Bend_Over")

            # ===== 4) 下蹲：膝盖明显升高接近髋部 =====
            knee_ok = self._visible(pose, [LEFT_KNEE, RIGHT_KNEE])
            if knee_ok:
                lk, rk = lm[LEFT_KNEE], lm[RIGHT_KNEE]
                knee_mid = _midpoint(lk, rk)
                body_h = max(np.linalg.norm((shoulder_mid - hip_mid)[:2]), 1e-6)
                knee_hip_dist = float(np.linalg.norm((knee_mid - hip_mid)[:2]))
                ratio = knee_hip_dist / body_h
                features["knee_hip_ratio"] = ratio
                # 站立时 knee 应远低于 hip，ratio 接近 1；下蹲时膝接近髋，ratio 变小
                if ratio < self.squat_knee_hip_ratio:
                    labels.append("Squat")

        # ===== 选主动作 =====
        if not labels:
            primary = "Standing"
            labels = ["Standing"]
        else:
            # 按优先级表选第一个出现的
            primary = next((a for a in ACTION_PRIORITY if a in labels), labels[0])

        return ActionResult(
            primary=primary, labels=labels, features=features, valid=True
        )
