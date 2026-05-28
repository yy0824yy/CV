"""
关键点 / 标量数据平滑工具。

提供三种平滑方式：
    1) EMASmoother           指数移动平均，对单个标量
    2) MovingAverageSmoother 滑动窗口均值
    3) LandmarkSmoother      对 (N, 2/3) 关键点数组做 EMA

以及一个用于离散类别（如手势/动作）稳定化的：
    4) MajorityVoter         最近 N 帧多数投票
"""
from __future__ import annotations

from collections import Counter, deque
from typing import Dict, Optional

import numpy as np


# ============================================================
# 1) 指数移动平均（EMA）：适合连续值如角度
# ============================================================
class EMASmoother:
    """指数移动平均平滑器。

    公式：y_t = alpha * x_t + (1 - alpha) * y_{t-1}

    alpha 越大越敏感（越接近原始值），越小越平滑（越滞后）。
    经验值：0.3~0.5 用于关节角度，0.5~0.8 用于关键点坐标。
    """

    def __init__(self, alpha: float = 0.4):
        self.alpha = float(alpha)
        self._y: Optional[float] = None

    def update(self, x: float) -> float:
        if x is None:
            return self._y if self._y is not None else 0.0
        if self._y is None:
            self._y = float(x)
        else:
            self._y = self.alpha * float(x) + (1.0 - self.alpha) * self._y
        return self._y

    def reset(self):
        self._y = None

    @property
    def value(self) -> Optional[float]:
        return self._y


# ============================================================
# 2) 滑动窗口均值
# ============================================================
class MovingAverageSmoother:
    """最近 N 帧的算术平均。"""

    def __init__(self, window: int = 5):
        self.window = int(window)
        self._buf: deque = deque(maxlen=self.window)

    def update(self, x: float) -> float:
        self._buf.append(float(x))
        return sum(self._buf) / len(self._buf)

    def reset(self):
        self._buf.clear()


# ============================================================
# 3) 关键点数组 EMA（对 (N, 2) 或 (N, 3) 整体平滑）
# ============================================================
class LandmarkSmoother:
    """对整张关键点数组做 EMA 平滑。

    用法：
        s = LandmarkSmoother(alpha=0.6)
        smoothed = s.update(raw_landmarks)   # raw shape=(33,3) 或 (21,3)

    若上一帧没有数据（None），自动初始化为当前值。
    若关键点数发生变化（不同尺寸），自动重置。
    """

    def __init__(self, alpha: float = 0.6):
        self.alpha = float(alpha)
        self._y: Optional[np.ndarray] = None

    def update(self, landmarks: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if landmarks is None:
            return self._y
        x = np.asarray(landmarks, dtype=np.float32)
        if self._y is None or self._y.shape != x.shape:
            self._y = x.copy()
        else:
            self._y = self.alpha * x + (1.0 - self.alpha) * self._y
        return self._y

    def reset(self):
        self._y = None


# ============================================================
# 4) 多数投票：用于离散类别（手势名 / 动作名）
# ============================================================
class MajorityVoter:
    """最近 N 帧的多数投票。

    用于让手势/动作识别结果不"跳"：
        识别器对每帧输出可能短暂抖动（A 帧识为 OK，下一帧识为 Number_3），
        该投票器返回最近 N 帧中出现次数最多的类别。
    """

    def __init__(self, window: int = 7):
        self.window = int(window)
        self._buf: deque = deque(maxlen=self.window)

    def update(self, label: str) -> str:
        self._buf.append(str(label))
        c = Counter(self._buf)
        return c.most_common(1)[0][0]

    @property
    def stable_label(self) -> Optional[str]:
        if not self._buf:
            return None
        return Counter(self._buf).most_common(1)[0][0]

    def reset(self):
        self._buf.clear()


# ============================================================
# 5) 加权多数投票：识别置信度作为权重
# ============================================================
class WeightedVoter:
    """加权多数投票。

    与简单 MajorityVoter 的区别：每帧投票携带"置信度权重"，
    最终输出的标签是 sum(权重) 最大的那一类，而不是出现次数最多的。

    场景：手势识别中某些帧识别置信度只有 0.6（边缘情况），
    与"非常确定的 0.95"应该差别对待。这样能更快从抖动恢复到正确标签。

    示例：
        v = WeightedVoter(window=7)
        v.update("OK", 0.9)
        v.update("Number_3", 0.5)
        v.update("OK", 0.95)
        v.stable_label  # -> "OK"  (因为 0.9+0.95=1.85 > 0.5)
    """

    def __init__(self, window: int = 7, min_score: float = 0.0):
        self.window = int(window)
        self.min_score = float(min_score)
        self._buf: deque = deque(maxlen=self.window)  # 每项 = (label, score)

    def update(self, label: str, score: float = 1.0) -> str:
        s = float(score)
        if s < self.min_score:
            # 太弱的预测不计入
            return self.stable_label or str(label)
        self._buf.append((str(label), s))
        return self.stable_label or str(label)

    @property
    def stable_label(self) -> Optional[str]:
        if not self._buf:
            return None
        weights: Dict[str, float] = {}
        counts: Dict[str, int] = {}
        for lbl, s in self._buf:
            weights[lbl] = weights.get(lbl, 0.0) + s
            counts[lbl] = counts.get(lbl, 0) + 1
        # 主排序：权重和；次排序：出现次数。权重先 round，避免
        # 0.3 * 3 这类浮点误差破坏理论上的平局。
        return max(
            weights.items(),
            key=lambda kv: (round(kv[1], 10), counts[kv[0]]),
        )[0]

    @property
    def stable_score(self) -> float:
        """当前胜出标签的平均置信度（用于上层显示）。"""
        if not self._buf:
            return 0.0
        win = self.stable_label
        same = [s for lbl, s in self._buf if lbl == win]
        return float(sum(same) / len(same)) if same else 0.0

    def reset(self):
        self._buf.clear()
