"""
活动日志记录器（应用层）。

功能：
    - 每帧从 ProcessedFrame 抽取"值得记录"的事件
    - 自动去重：连续多帧相同状态合并为一个事件
    - 维护时间窗口内的事件流，供 AI 日报使用

记录的事件类型（type 字段）：
    - 'action_change'      : 主动作切换（如 站立 -> 举右手）
    - 'dynamic_gesture'    : 动态手势触发（Grab / Pinch / Release / Point）
    - 'fall'               : 跌倒事件
    - 'distance_warning'   : 距离过近 / 过远进入或解除
    - 'session_start'      : 开机
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Deque, List, Optional

if TYPE_CHECKING:
    from ui.video_thread import ProcessedFrame


# 动作 / 手势的中文映射（与 llm_understand 保持一致；这里独立定义避免循环引用）
_ACTION_CN = {
    "Standing":         "站立",
    "Raise_Left_Hand":  "举左手",
    "Raise_Right_Hand": "举右手",
    "Raise_Both_Hands": "举双手",
    "Lean_Left":        "向左倾",
    "Lean_Right":       "向右倾",
    "Squat":            "下蹲",
    "Bend_Forward":     "弯腰",
    "No_Person":        "未在画面中",
    "Unknown":          "动作不明",
}

_DYN_CN = {
    "Grab":    "握拳（Grab）",
    "Release": "张手（Release）",
    "Pinch":   "OK 手势（Pinch）",
    "Point":   "指向（Point）",
}


@dataclass
class ActivityEvent:
    """一条活动事件记录。"""
    timestamp: float
    type: str
    description: str
    metadata: dict = field(default_factory=dict)

    def time_str(self) -> str:
        return time.strftime("%H:%M:%S", time.localtime(self.timestamp))


class ActivityRecorder:
    """
    用法：
        rec = ActivityRecorder()
        rec.session_start()
        # 每帧:
        new_events = rec.update(processed_frame)
        # 想要 AI 日报:
        events = rec.recent_events(since_seconds=300)
    """

    def __init__(
        self,
        max_events: int = 500,
        max_age_seconds: float = 1800.0,
        # 同一动作至少持续 X 秒才视为"实质性的动作切换"，避免抖动刷条
        action_min_duration_sec: float = 1.0,
    ):
        self._events: Deque[ActivityEvent] = deque(maxlen=max_events)
        self._max_age_seconds = float(max_age_seconds)
        self._action_min_duration_sec = float(action_min_duration_sec)

        # 状态记忆：用于去重 / 检测变化
        self._last_action: Optional[str] = None
        self._last_action_since: float = 0.0
        self._pending_action: Optional[str] = None      # 候选新动作
        self._pending_since: float = 0.0
        self._last_dynamic_ts: float = 0.0
        self._last_close_state: bool = False
        self._last_far_state: bool = False
        self._last_fall_ts: float = 0.0

    # ------------- 显式接口 -------------
    def session_start(self):
        self._append(ActivityEvent(
            timestamp=time.time(),
            type="session_start",
            description="会话开始：系统已上线，开始监测用户活动",
        ))

    def clear(self):
        self._events.clear()
        self._last_action = None
        self._last_action_since = 0.0
        self._pending_action = None
        self._pending_since = 0.0
        self._last_dynamic_ts = 0.0
        self._last_close_state = False
        self._last_far_state = False
        self._last_fall_ts = 0.0

    # ------------- 主入口 -------------
    def update(self, processed: "ProcessedFrame") -> List[ActivityEvent]:
        """根据当前帧抽取新事件（已去重），返回本次新增的事件列表。"""
        new_events: List[ActivityEvent] = []
        now = time.time()

        # ---- 1) 动作切换（带最小持续时长去抖） ----
        action_label = None
        if processed.action and processed.action.valid:
            action_label = processed.action.primary
        if action_label and action_label not in ("Unknown",):
            if action_label != self._last_action:
                # 候选新动作：必须持续 N 秒才能确认
                if action_label != self._pending_action:
                    self._pending_action = action_label
                    self._pending_since = now
                elif (now - self._pending_since) >= self._action_min_duration_sec:
                    # 确认切换
                    cn = _ACTION_CN.get(action_label, action_label)
                    desc = f"动作切换：{cn}"
                    if self._last_action is not None:
                        prev_cn = _ACTION_CN.get(
                            self._last_action, self._last_action)
                        desc = f"动作切换：{prev_cn} → {cn}"
                    ev = ActivityEvent(
                        timestamp=now, type="action_change",
                        description=desc,
                        metadata={"from": self._last_action, "to": action_label},
                    )
                    self._append(ev)
                    new_events.append(ev)
                    self._last_action = action_label
                    self._last_action_since = now
                    self._pending_action = None
            else:
                self._pending_action = None  # 没变化

        # ---- 2) 动态手势事件 ----
        dyn = processed.dynamic_gesture
        if dyn is not None and dyn.name and (now - self._last_dynamic_ts) > 0.3:
            cn = _DYN_CN.get(dyn.name, dyn.name)
            ev = ActivityEvent(
                timestamp=now, type="dynamic_gesture",
                description=f"动态手势：{cn}",
                metadata={"name": dyn.name},
            )
            self._append(ev)
            new_events.append(ev)
            self._last_dynamic_ts = now

        # ---- 3) 跌倒事件 ----
        if processed.fall_event is not None:
            ev = processed.fall_event
            new = ActivityEvent(
                timestamp=ev.timestamp, type="fall",
                description=(
                    f"⚠️ 检测到跌倒（躯干倾角 {ev.torso_angle_deg:.0f}°，"
                    f"持续 {ev.confirmed_duration_sec:.1f}s）"
                ),
                metadata={
                    "torso_angle_deg": ev.torso_angle_deg,
                    "confirmed_duration_sec": ev.confirmed_duration_sec,
                    "person_distance_m": ev.person_distance_m,
                },
            )
            self._append(new)
            new_events.append(new)
            self._last_fall_ts = ev.timestamp

        # ---- 4) 距离异常状态切换（进入 / 解除） ----
        if processed.too_close != self._last_close_state:
            self._last_close_state = processed.too_close
            if processed.too_close:
                desc = (
                    f"距离过近：{processed.person_distance_m:.2f}m"
                    if processed.person_distance_m else "距离过近"
                )
                ev = ActivityEvent(
                    timestamp=now, type="distance_warning",
                    description=desc,
                    metadata={"state": "too_close"},
                )
                self._append(ev)
                new_events.append(ev)
        if processed.too_far != self._last_far_state:
            self._last_far_state = processed.too_far
            if processed.too_far:
                desc = (
                    f"距离过远：{processed.person_distance_m:.2f}m"
                    if processed.person_distance_m else "距离过远"
                )
                ev = ActivityEvent(
                    timestamp=now, type="distance_warning",
                    description=desc,
                    metadata={"state": "too_far"},
                )
                self._append(ev)
                new_events.append(ev)

        # ---- 5) 顺手清理过老事件（保留最近 max_age_seconds 内的） ----
        cutoff = now - self._max_age_seconds
        while self._events and self._events[0].timestamp < cutoff:
            self._events.popleft()

        return new_events

    # ------------- 查询 -------------
    def recent_events(self, since_seconds: float = 300.0) -> List[ActivityEvent]:
        cutoff = time.time() - since_seconds
        return [e for e in self._events if e.timestamp >= cutoff]

    def all_events(self) -> List[ActivityEvent]:
        return list(self._events)

    def total_count(self) -> int:
        return len(self._events)

    # ------------- 工具 -------------
    def _append(self, ev: ActivityEvent):
        self._events.append(ev)
