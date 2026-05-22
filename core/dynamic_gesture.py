"""
动态手势识别（状态切换型）。

设计：
    所有 4 种动态手势都基于 "前一个静态手势 -> 后一个静态手势" 的切换检测。
    优点：依赖已经稳定的静态手势识别（ML 准确率 100%），触发率高、误触发少。

支持识别（在 0.6 秒窗口内完成）：
    - Grab    抓取 ：Open_Palm -> Fist
    - Release 释放 ：Fist -> Open_Palm
    - Pinch   捏合 ：Open_Palm -> OK
    - Point   指向 ：Open_Palm -> Number_1

实现要点：
    - 维护时间窗口缓冲：每帧追加 (t, wrist_x, wrist_y, static_gesture)
    - 检测条件：缓冲前半段含 "from" 手势 + 最后一帧是 "to" 手势
    - 事件冷却 1s 防止连续触发；fade 2s 用于 UI 显示
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Tuple

import numpy as np


# ============================================================
# 4 种状态切换：(事件名, from 手势, to 手势)
# ============================================================
TRANSITIONS: List[Tuple[str, str, str]] = [
    ("Grab",    "Open_Palm", "Fist"),
    ("Release", "Fist",      "Open_Palm"),
    ("Pinch",   "Open_Palm", "OK"),
    ("Point",   "Open_Palm", "Number_1"),
]


@dataclass
class DynamicGestureResult:
    name: str          # Grab / Release / Pinch / Point
    score: float       # 触发"强度"占位（这里固定 1.0，可扩展为切换速度等）
    timestamp: float


# 中文名映射（UI 用）
DYNAMIC_GESTURE_CN = {
    "Grab":    "抓取",
    "Release": "释放",
    "Pinch":   "捏合",
    "Point":   "指向",
}


# ============================================================
# 识别器
# ============================================================
class DynamicGestureRecognizer:
    """单只手的动态手势识别器（状态切换型）。"""

    # 切换窗口长度（秒）：from -> to 必须在这段时间内完成
    WINDOW_SECONDS = 0.6
    # 缓冲区最大保留时长
    HISTORY_SECONDS = 1.0

    def __init__(self, cooldown_seconds: float = 1.0,
                 fade_seconds: float = 2.0):
        self.cooldown = cooldown_seconds
        self.fade_seconds = fade_seconds
        # (timestamp, wrist_x_norm, wrist_y_norm, static_gesture_name)
        self._buf: Deque[Tuple[float, float, float, str]] = deque()
        self._last_event_t = 0.0
        self._last_event: Optional[DynamicGestureResult] = None
        # 已触发后必须先 "离开" to 手势才能再次触发，防止持续保持时连发
        self._last_to: Optional[str] = None

    # ------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------
    def update(self, wrist_x: Optional[float], wrist_y: Optional[float],
               static_gesture: str,
               timestamp: Optional[float] = None
               ) -> Optional[DynamicGestureResult]:
        """每帧调用一次。返回新触发的事件 或 None。"""
        if timestamp is None:
            timestamp = time.time()

        # 没手则只清空（不在每帧都被调用导致缓冲被清空，调用方自己控制）
        if wrist_x is None or wrist_y is None:
            self._buf.clear()
            self._last_to = None
            return None

        self._buf.append((timestamp, float(wrist_x), float(wrist_y),
                          str(static_gesture or "")))
        # 丢弃过期项
        while self._buf and timestamp - self._buf[0][0] > self.HISTORY_SECONDS:
            self._buf.popleft()

        # 当用户离开上一次的 "to" 手势后，重置抑制
        if self._last_to is not None and static_gesture != self._last_to:
            self._last_to = None

        # 冷却中
        if timestamp - self._last_event_t < self.cooldown:
            return None
        if len(self._buf) < 6:
            return None

        ev = self._detect_transition(timestamp)
        if ev is not None:
            self._last_event_t = timestamp
            self._last_event = ev
            # 找到该事件的 to 手势记录下来防止重发
            for name, _f, t_g in TRANSITIONS:
                if name == ev.name:
                    self._last_to = t_g
                    break
            return ev
        return None

    def get_last_event(self, now: Optional[float] = None
                       ) -> Optional[DynamicGestureResult]:
        """获取最近一次事件，超过 fade_seconds 返回 None。"""
        if self._last_event is None:
            return None
        now = now if now is not None else time.time()
        if now - self._last_event.timestamp > self.fade_seconds:
            return None
        return self._last_event

    def reset(self):
        self._buf.clear()
        self._last_event = None
        self._last_event_t = 0.0
        self._last_to = None

    # ------------------------------------------------------------
    # 状态切换检测
    # ------------------------------------------------------------
    def _detect_transition(self, t_now: float
                           ) -> Optional[DynamicGestureResult]:
        """
        在最近 WINDOW_SECONDS 秒内查找：
            - 前半段（时间上较早的帧）里出现过某个 "from" 手势
            - 最后一帧（current）是对应的 "to" 手势
            - 抑制：to 手势没变化（_last_to 还在）则不触发
        """
        recent = [(t, g) for (t, _x, _y, g) in self._buf
                  if t_now - t <= self.WINDOW_SECONDS]
        if len(recent) < 6:
            return None
        gestures = [g for _, g in recent]
        n = len(gestures)
        early = set(gestures[: n // 2])
        last = gestures[-1]

        for name, from_g, to_g in TRANSITIONS:
            if last != to_g:
                continue
            if from_g not in early:
                continue
            # 抑制：刚刚已经触发过同一个 to_g 且用户还没离开
            if self._last_to == to_g:
                continue
            return DynamicGestureResult(name, 1.0, t_now)
        return None
