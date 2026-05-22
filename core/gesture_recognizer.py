"""
基于 21 关键点的手势识别（纯规则方法，不训练模型）。

算法核心：
    1. 判定每根手指是"伸直"还是"弯曲"（finger_states）
    2. 计算辅助特征：拇指-食指距离、拇指朝向、手心朝向
    3. 按规则字典匹配，最高分胜出

设计目标：
    - 不依赖训练数据，纯几何规则，可解释性强（适合写报告/讲答辩）
    - 输出结构化结果 GestureResult，包含手势名、置信度、调试信息

支持手势（8 类 + Unknown）：
    Number_1, Number_2, Number_3, Number_5/Open_Palm,
    Fist, OK, Thumbs_Up, Unknown
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from core.hand_detector import (
    HandLandmarks,
    FINGER_TIPS,
    FINGER_PIPS,
    FINGER_MCPS,
)


# ============================================================
# 数据结构
# ============================================================
@dataclass
class GestureResult:
    """一只手的手势识别结果。"""
    name: str                                       # 手势名（如 "Number_1"）
    score: float                                    # 匹配分数 [0,1]
    finger_states: List[bool] = field(default_factory=list)  # [拇,食,中,无名,小] True=伸直
    extra: Dict[str, float] = field(default_factory=dict)    # 调试用特征值


# ============================================================
# 几何工具
# ============================================================
def _dist_norm(p1: np.ndarray, p2: np.ndarray) -> float:
    """归一化坐标下两点距离（与图像尺寸无关）。"""
    return float(np.linalg.norm(p1[:2] - p2[:2]))


def _palm_size(landmarks_norm: np.ndarray) -> float:
    """估算手掌尺寸（用 wrist→middle_mcp 距离）。
    用于把其他距离归一化为"相对手掌大小"的比例，
    使阈值不随手离镜头远近变化。
    """
    wrist = landmarks_norm[0]
    middle_mcp = landmarks_norm[9]
    return max(_dist_norm(wrist, middle_mcp), 1e-6)


# ============================================================
# 手指伸直判定
# ============================================================
def _finger_extended(
    landmarks_norm: np.ndarray,
    finger_idx: int,
    handedness: str,
    palm_up: bool,
) -> bool:
    """判断某根手指是否伸直（朝向鲁棒版）。

    核心思想：手指弯曲时，指尖必然向手腕方向卷回，因此
        伸直 ⇔ dist(tip, wrist) > dist(pip, wrist)
    这个几何关系与手心/手背朝向无关，也不依赖 y 坐标，
    比单纯比较 y 更鲁棒。

    Args:
        landmarks_norm: 21x3 归一化坐标
        finger_idx: 0=拇指, 1=食指, 2=中指, 3=无名指, 4=小指
        handedness: 保留参数，仅作日志用
        palm_up: 保留参数，仅作日志用
    """
    tip = landmarks_norm[FINGER_TIPS[finger_idx]]
    pip = landmarks_norm[FINGER_PIPS[finger_idx]]

    if finger_idx == 0:
        # 拇指：参考点用食指根 INDEX_FINGER_MCP(5)
        # 拇指弯曲时指尖会贴向手心、靠近食指根；伸直时远离
        ref = landmarks_norm[5]
        # 加一点裕度：拇指要明显远离才算伸直，避免抖动
        return _dist_norm(tip, ref) > _dist_norm(pip, ref) * 1.1
    else:
        # 四指：参考点用 wrist(0)
        wrist = landmarks_norm[0]
        return _dist_norm(tip, wrist) > _dist_norm(pip, wrist)


def _is_palm_up(landmarks_norm: np.ndarray) -> bool:
    """判断手心/手指朝向。

    简化版：用 middle_finger_mcp(9) 与 wrist(0) 的 y 关系：
    - middle_mcp.y < wrist.y → 手指朝上（"palm_up"）
    - 反之 → 手指朝下
    """
    return landmarks_norm[9][1] < landmarks_norm[0][1]


# ============================================================
# 手势识别主类
# ============================================================
class GestureRecognizer:
    """基于规则的手势识别器。"""

    def __init__(self, min_score: float = 0.6):
        self.min_score = min_score

    def recognize(self, hand: HandLandmarks) -> GestureResult:
        """识别一只手的手势。"""
        lm = hand.landmarks_norm
        palm_up = _is_palm_up(lm)

        # 1) 每根手指伸直状态
        states = [
            _finger_extended(lm, i, hand.handedness, palm_up)
            for i in range(5)
        ]

        # 2) 辅助特征
        palm = _palm_size(lm)
        thumb_index_dist = _dist_norm(lm[4], lm[8]) / palm  # 相对手掌尺寸的拇-食距离
        # 拇指朝向（向上 = thumb_tip.y < wrist.y 较多）
        thumb_up_ratio = (lm[0][1] - lm[4][1]) / palm   # 越大越朝上

        extra = {
            "thumb_index_dist": thumb_index_dist,
            "thumb_up_ratio": thumb_up_ratio,
            "palm_up": float(palm_up),
        }

        # 3) 规则匹配
        candidates: List[Tuple[str, float]] = []

        # --- 数字手势 ---
        # 注意：数字 1/2/3 仅约束四指（食/中/无名/小）的伸直模式，
        # 拇指状态不限。原因：手背朝向时拇指 2D 检测易抖动误判为伸直，
        # 且现实中比数字时拇指自然位置因人而异，不应严格要求弯曲。
        four = states[1:]   # [食, 中, 无名, 小]
        if four == [True, False, False, False]:
            candidates.append(("Number_1", 1.0))
        if four == [True, True, False, False]:
            candidates.append(("Number_2", 1.0))
        if four == [True, True, True, False]:
            candidates.append(("Number_3", 1.0))
        # Open_Palm / Number_5：五指全伸直（必须包含拇指）
        if states == [True, True, True, True, True]:
            candidates.append(("Open_Palm", 1.0))

        # --- 握拳：五指都弯曲 ---
        if states == [False, False, False, False, False]:
            candidates.append(("Fist", 1.0))

        # --- OK 手势：食指与拇指捏成圈（食指必须弯曲与拇指相接），中/无名/小指伸直 ---
        # 关键约束：食指必须"非伸直"，否则会把 Number_3（食中无名伸直）误判为 OK
        # 经验阈值：thumb_index_dist < 0.5（相对手掌尺寸）
        ok_close = thumb_index_dist < 0.5
        if ok_close and (not states[1]) and states[2] and states[3] and states[4]:
            # 给一个连续分数：距离越近分越高
            score = 1.0 - min(thumb_index_dist / 0.5, 1.0) * 0.4   # 0.6~1.0
            candidates.append(("OK", score))

        # --- 点赞：仅拇指伸直，其他四指弯曲，且拇指明显朝上 ---
        if states == [True, False, False, False, False] and thumb_up_ratio > 0.6:
            score = min(thumb_up_ratio / 1.0, 1.0)
            candidates.append(("Thumbs_Up", max(score, 0.7)))

        # 4) 选最高分；若无候选则 Unknown
        if not candidates:
            return GestureResult(
                name="Unknown",
                score=0.0,
                finger_states=states,
                extra=extra,
            )

        name, score = max(candidates, key=lambda x: x[1])
        if score < self.min_score:
            name = "Unknown"
        return GestureResult(
            name=name, score=score, finger_states=states, extra=extra
        )

    def recognize_all(self, hands: List[HandLandmarks]) -> List[GestureResult]:
        """批量识别多只手。"""
        return [self.recognize(h) for h in hands]
