"""
带连续帧多数投票的稳定识别器（高层包装）。

包装策略：
    StableGestureRecognizer = GestureRecognizer + per-hand MajorityVoter
    StableActionRecognizer  = ActionRecognizer  + MajorityVoter

向上层提供与原识别器一致的接口，但输出是经过最近 N 帧投票后的稳定标签，
有效抑制单帧跳变（如 OK 偶尔识别为 Number_3）。
"""
from __future__ import annotations

from dataclasses import replace
from typing import Dict, List, Optional

from core.hand_detector import HandLandmarks
from core.pose_detector import PoseLandmarks
from core.gesture_recognizer import GestureRecognizer, GestureResult
from core.action_recognizer import ActionRecognizer, ActionResult
from core.smoothing import MajorityVoter, WeightedVoter


class StableGestureRecognizer:
    """对 GestureRecognizer 包一层加权多数投票。

    每只手（Left / Right）独立维护一个 voter，互不影响。
    使用 WeightedVoter：每帧投票携带置信度权重，能更快从抖动中
    恢复到正确标签（高置信度的预测压过低置信度的）。
    """

    def __init__(self, base: Optional[GestureRecognizer] = None,
                 window: int = 7, min_score: float = 0.4):
        self._base = base or GestureRecognizer()
        self._window = int(window)
        self._min_score = float(min_score)
        self._voters: Dict[str, WeightedVoter] = {}

    def _get_voter(self, key: str) -> WeightedVoter:
        if key not in self._voters:
            self._voters[key] = WeightedVoter(
                window=self._window, min_score=self._min_score
            )
        return self._voters[key]

    def recognize(self, hand: HandLandmarks) -> GestureResult:
        """识别一只手并稳定化。"""
        raw = self._base.recognize(hand)
        voter = self._get_voter(hand.handedness)
        stable = voter.update(raw.name, raw.score)
        # 输出置信度：使用胜出标签的平均权重（投票稳定性指标）
        score = voter.stable_score if stable == raw.name else min(raw.score, 0.7)
        return replace(raw, name=stable, score=score)

    def recognize_all(self, hands: List[HandLandmarks]) -> List[GestureResult]:
        return [self.recognize(h) for h in hands]

    def reset(self):
        for v in self._voters.values():
            v.reset()


class StableActionRecognizer:
    """对 ActionRecognizer 包一层多数投票。"""

    def __init__(self, base: Optional[ActionRecognizer] = None, window: int = 9):
        self._base = base or ActionRecognizer()
        self._voter = MajorityVoter(window=int(window))

    def recognize(self, pose: Optional[PoseLandmarks]) -> ActionResult:
        raw = self._base.recognize(pose)
        stable = self._voter.update(raw.primary)
        return replace(raw, primary=stable)

    def reset(self):
        self._voter.reset()
