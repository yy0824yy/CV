"""GestureTrigger 的单元测试。

覆盖触发器的关键行为：
    - 持续 N 帧才触发
    - 冷却期内不重复触发
    - 触发后必须先松手才能再触发（关键 UX 设计）
    - 多手独立维护
    - 触发表外的手势不触发
"""
import time
from unittest.mock import patch

import pytest

from core.gesture_trigger import GestureTrigger, TriggerEvent, DEFAULT_TRIGGER_MAP


# ============================================================
# 工具：fake time
# ============================================================
class FakeClock:
    """可手动推进的虚拟时钟，用 patch 替换 time.time。"""
    def __init__(self, t0: float = 1000.0):
        self.t = t0
    def __call__(self):
        return self.t
    def advance(self, sec: float):
        self.t += sec


# ============================================================
# 基本触发
# ============================================================
class TestGestureTriggerBasic:
    def test_no_trigger_below_min_hold(self):
        """少于 min_hold_frames 帧不触发。"""
        trig = GestureTrigger(min_hold_frames=5)
        for _ in range(4):
            evt = trig.update("Right", "OK")
            assert evt is None

    def test_fires_after_min_hold_frames(self):
        trig = GestureTrigger(min_hold_frames=5)
        for _ in range(4):
            assert trig.update("Right", "OK") is None
        evt = trig.update("Right", "OK")
        assert evt is not None
        assert evt.action == "snapshot"  # OK -> snapshot
        assert evt.gesture == "OK"
        assert evt.handedness == "Right"

    def test_default_map_contains_three_actions(self):
        assert DEFAULT_TRIGGER_MAP["OK"] == "snapshot"
        assert DEFAULT_TRIGGER_MAP["Fist"] == "toggle_record"
        assert DEFAULT_TRIGGER_MAP["Thumbs_Up"] == "toggle_pause"

    def test_unmapped_gesture_does_not_fire(self):
        trig = GestureTrigger(min_hold_frames=3)
        for _ in range(10):
            evt = trig.update("Right", "Number_3")
            assert evt is None  # Number_3 不在 trigger_map 中

    def test_unknown_gesture_breaks_accumulation(self):
        """中间穿插非触发手势会打断累计。"""
        trig = GestureTrigger(min_hold_frames=5)
        # 累 4 帧 OK，第 5 帧手势变 Unknown，再切回 OK 时应重新累计
        for _ in range(4):
            trig.update("Right", "OK")
        trig.update("Right", "Unknown")
        # 此时只有 1 帧 OK 在最新位置
        for _ in range(3):
            evt = trig.update("Right", "OK")
            assert evt is None
        # 再 1 帧补满 5 帧
        evt = trig.update("Right", "OK")
        assert evt is not None


# ============================================================
# 冷却期
# ============================================================
class TestCooldown:
    def test_cooldown_blocks_immediate_refire(self):
        clock = FakeClock()
        with patch("core.gesture_trigger.time.time", clock):
            trig = GestureTrigger(min_hold_frames=3, cooldown_sec=2.0)
            for _ in range(3):
                pass
            # 累计触发一次
            for _ in range(2):
                trig.update("Right", "OK")
            evt1 = trig.update("Right", "OK")
            assert evt1 is not None

            # 持续按住 OK，期间不应再次触发（这是关键 UX）
            for _ in range(20):
                evt = trig.update("Right", "OK")
                assert evt is None

            # 即使时间走过 cooldown，只要不松手仍不触发
            clock.advance(3.0)
            for _ in range(20):
                evt = trig.update("Right", "OK")
                assert evt is None

    def test_release_then_re_hold_fires_again(self):
        """松手 → 重新做手势 → 应该能再次触发。"""
        clock = FakeClock()
        with patch("core.gesture_trigger.time.time", clock):
            trig = GestureTrigger(min_hold_frames=3, cooldown_sec=1.0)
            # 第一次触发
            for _ in range(3):
                trig.update("Right", "OK")
            assert trig._last_fire  # 已触发过
            # 松手（穿插非触发手势）
            for _ in range(5):
                trig.update("Right", "Unknown")
            # 等待冷却
            clock.advance(1.5)
            # 重做 OK，应能触发
            for _ in range(2):
                trig.update("Right", "OK")
            evt = trig.update("Right", "OK")
            assert evt is not None
            assert evt.action == "snapshot"


# ============================================================
# 多手独立
# ============================================================
class TestPerHand:
    def test_left_and_right_independent(self):
        trig = GestureTrigger(min_hold_frames=3)
        # 右手累 2 帧 OK
        trig.update("Right", "OK")
        trig.update("Right", "OK")
        # 左手累 3 帧 OK，应触发
        trig.update("Left", "OK")
        trig.update("Left", "OK")
        evt = trig.update("Left", "OK")
        assert evt is not None
        assert evt.handedness == "Left"

    def test_right_hand_unaffected_by_left_trigger(self):
        clock = FakeClock()
        with patch("core.gesture_trigger.time.time", clock):
            trig = GestureTrigger(min_hold_frames=3, cooldown_sec=2.0)
            # 左手触发
            for _ in range(3):
                trig.update("Left", "OK")
            # 右手在同一时间触发，不应被左手的冷却影响
            for _ in range(2):
                trig.update("Right", "OK")
            evt = trig.update("Right", "OK")
            assert evt is not None
            assert evt.handedness == "Right"


# ============================================================
# 自定义映射
# ============================================================
class TestCustomMap:
    def test_custom_trigger_map(self):
        trig = GestureTrigger(
            trigger_map={"Number_5": "wave_hello"},
            min_hold_frames=3,
        )
        # OK 不在自定义表里，不触发
        for _ in range(10):
            assert trig.update("Right", "OK") is None
        # Number_5 在表里
        for _ in range(2):
            trig.update("Right", "Number_5")
        evt = trig.update("Right", "Number_5")
        assert evt is not None
        assert evt.action == "wave_hello"


# ============================================================
# Reset
# ============================================================
class TestReset:
    def test_reset_clears_state(self):
        clock = FakeClock()
        with patch("core.gesture_trigger.time.time", clock):
            trig = GestureTrigger(min_hold_frames=3, cooldown_sec=1.0)
            for _ in range(3):
                trig.update("Right", "OK")  # 触发过
            assert trig._last_fire
            trig.reset()
            assert not trig._last_fire
            # 立即重做 3 帧应能再触发（无残留冷却）
            for _ in range(2):
                trig.update("Right", "OK")
            evt = trig.update("Right", "OK")
            assert evt is not None
