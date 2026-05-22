"""
手势触发器：检测稳定持续的特定手势，自动触发回调动作。

设计目标：
    用户做一个手势 N 帧后，自动触发一次"动作"（如截图、录制开关），
    然后进入冷却期避免重复触发。

工作流程：
    1) 每帧调用 update(handedness, gesture_name)
    2) 当某只手持续 N 帧识别为目标手势时，触发对应回调
    3) 触发后进入 cooldown 期（数秒），期间该手势不再触发

支持手势 → 动作映射（默认）：
    OK         → snapshot      （拍照）
    Fist       → toggle_record （录像开关）
    Thumbs_Up  → toggle_pause  （暂停/继续）
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Dict, Optional


# 默认触发映射：gesture_name -> action_key
DEFAULT_TRIGGER_MAP: Dict[str, str] = {
    "OK": "snapshot",
    "Fist": "toggle_record",
    "Thumbs_Up": "toggle_pause",
}


@dataclass
class TriggerEvent:
    """触发事件，用于通知 UI。"""
    action: str            # 动作 key，如 "snapshot"
    gesture: str           # 触发的手势名
    handedness: str        # 哪只手
    timestamp: float       # 触发时间


class GestureTrigger:
    """稳定手势 -> 动作触发器。

    使用方法：
        trigger = GestureTrigger(
            trigger_map={"OK": "snapshot", "Fist": "toggle_record"},
            min_hold_frames=12,    # 大约 1 秒（取决于 FPS）
            cooldown_sec=3.0,      # 触发后 3 秒冷却
        )
        # 每帧：
        evt = trigger.update("Right", "OK")
        if evt is not None:
            handle_action(evt.action)
    """

    def __init__(
        self,
        trigger_map: Optional[Dict[str, str]] = None,
        min_hold_frames: int = 12,
        cooldown_sec: float = 3.0,
        history: int = 30,
    ):
        self.trigger_map = dict(trigger_map or DEFAULT_TRIGGER_MAP)
        self.min_hold_frames = int(min_hold_frames)
        self.cooldown_sec = float(cooldown_sec)
        # 每只手维护一个最近 history 帧的手势序列
        self._buf: Dict[str, deque] = {}
        self._history = int(history)
        # 每个 (handedness, gesture) 上次触发时间
        self._last_fire: Dict[str, float] = {}
        # 每只手"未松手"的手势：触发后保持此手势仍不会再触发，
        # 必须看到不同手势（松手/换手势）后才会解除该锁
        self._held: Dict[str, str] = {}

    def _get_buf(self, key: str) -> deque:
        if key not in self._buf:
            self._buf[key] = deque(maxlen=self._history)
        return self._buf[key]

    def update(self, handedness: str, gesture: str) -> Optional[TriggerEvent]:
        """每帧调用一次。返回触发事件或 None。

        关键 UX 设计 ("must release")：
            触发后会给该手设置一个"锁"，锁定的手势即使继续保持也不会再触发；
            必须先识别到"不同的手势"（即用户松手或换手势）才解锁。
            这样彻底避免长按手势导致重复触发。
        """
        if not gesture:
            gesture = "Unknown"
        buf = self._get_buf(handedness)

        # 1) 若该手处于"锁定"状态：必须看到不同手势才解锁
        held = self._held.get(handedness)
        if held is not None:
            if gesture != held:
                # "松手"了，解锁；本帧不触发，并从新手势开始累计
                self._held.pop(handedness, None)
                buf.clear()
                buf.append(gesture)
            # 仍处于锁定（或刚解锁），本帧都不触发
            return None

        # 2) 非触发手势：作为"松手帧"加入 buffer，能打断之前的累计
        if gesture not in self.trigger_map:
            buf.append(gesture)
            return None

        # 3) 在冷却期内：忽略（冷却防止极短时间内同动作误触发）
        fire_key = f"{handedness}|{gesture}"
        now = time.time()
        last = self._last_fire.get(fire_key, 0.0)
        if (now - last) < self.cooldown_sec:
            return None

        # 4) 累计帧
        buf.append(gesture)
        if len(buf) < self.min_hold_frames:
            return None
        recent = list(buf)[-self.min_hold_frames:]
        if any(g != gesture for g in recent):
            return None

        # 5) 触发！设置 "held" 锁，更新冷却时间，清空累计
        self._last_fire[fire_key] = now
        self._held[handedness] = gesture
        buf.clear()
        return TriggerEvent(
            action=self.trigger_map[gesture],
            gesture=gesture,
            handedness=handedness,
            timestamp=now,
        )

    def reset(self):
        for b in self._buf.values():
            b.clear()
        self._last_fire.clear()
        self._held.clear()

    def cooldown_remaining(self, handedness: str, gesture: str) -> float:
        """剩余冷却时间，UI 可以用来画进度条。"""
        fire_key = f"{handedness}|{gesture}"
        last = self._last_fire.get(fire_key, 0.0)
        return max(0.0, self.cooldown_sec - (time.time() - last))
