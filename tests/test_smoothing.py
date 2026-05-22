"""smoothing 模块的单元测试。

覆盖：
    - EMASmoother
    - MajorityVoter
    - WeightedVoter（重点：今天新增）
"""
import math

import pytest

from core.smoothing import EMASmoother, MajorityVoter, WeightedVoter


# ============================================================
# EMASmoother
# ============================================================
class TestEMASmoother:
    def test_first_value_initializes_state(self):
        s = EMASmoother(alpha=0.5)
        assert s.update(10.0) == 10.0

    def test_alpha_05_averages(self):
        s = EMASmoother(alpha=0.5)
        s.update(0.0)
        # y = 0.5 * 10 + 0.5 * 0 = 5.0
        assert s.update(10.0) == pytest.approx(5.0)

    def test_alpha_1_returns_input(self):
        """alpha=1 时无平滑，等于原始值。"""
        s = EMASmoother(alpha=1.0)
        s.update(0.0)
        assert s.update(7.0) == pytest.approx(7.0)
        assert s.update(-3.0) == pytest.approx(-3.0)

    def test_alpha_0_locks_first_value(self):
        """alpha=0 极端情况只保留首值。"""
        s = EMASmoother(alpha=0.0)
        s.update(5.0)
        assert s.update(100.0) == pytest.approx(5.0)
        assert s.update(-100.0) == pytest.approx(5.0)

    def test_none_input_returns_state(self):
        s = EMASmoother(alpha=0.5)
        assert s.update(None) == 0.0  # 未初始化时返回 0
        s.update(10.0)
        assert s.update(None) == pytest.approx(10.0)

    def test_reset_clears_state(self):
        s = EMASmoother(alpha=0.5)
        s.update(10.0)
        s.reset()
        # 重置后第一次更新应该等于输入（重新初始化）
        assert s.update(50.0) == pytest.approx(50.0)


# ============================================================
# MajorityVoter
# ============================================================
class TestMajorityVoter:
    def test_single_label_wins(self):
        v = MajorityVoter(window=5)
        for _ in range(3):
            v.update("A")
        assert v.stable_label == "A"

    def test_majority_over_minority(self):
        v = MajorityVoter(window=5)
        for _ in range(3):
            v.update("A")
        for _ in range(2):
            v.update("B")
        # 5 帧中 A 出现 3 次，B 出现 2 次
        assert v.stable_label == "A"

    def test_window_evicts_old(self):
        """超过窗口的旧票被淘汰。"""
        v = MajorityVoter(window=3)
        v.update("A")
        v.update("B")
        v.update("B")
        v.update("B")
        # 窗口内现在是 B B B（A 已被挤出）
        assert v.stable_label == "B"

    def test_empty_buffer_returns_none(self):
        v = MajorityVoter(window=5)
        assert v.stable_label is None

    def test_reset(self):
        v = MajorityVoter(window=5)
        v.update("A")
        v.reset()
        assert v.stable_label is None


# ============================================================
# WeightedVoter（今天新增的核心组件）
# ============================================================
class TestWeightedVoter:
    def test_high_score_dominates_low_score(self):
        """两次高置信 OK 应胜过一次低置信 Number_3。"""
        v = WeightedVoter(window=5)
        v.update("OK", 0.9)
        v.update("Number_3", 0.5)
        v.update("OK", 0.95)
        # OK 权重和 = 1.85；Number_3 = 0.5
        assert v.stable_label == "OK"

    def test_count_breaks_tie_when_weights_equal(self):
        """权重相同时，按出现次数解平局。"""
        v = WeightedVoter(window=10)
        # A: 三次 0.3 = 0.9
        # B: 一次 0.9   = 0.9
        for _ in range(3):
            v.update("A", 0.3)
        v.update("B", 0.9)
        # 权重相等 → 看次数：A 3 > B 1
        assert v.stable_label == "A"

    def test_min_score_filters_weak_predictions(self):
        """min_score 阈值以下的预测不计入。"""
        v = WeightedVoter(window=5, min_score=0.5)
        v.update("Noise", 0.1)
        v.update("Noise", 0.2)
        v.update("OK", 0.95)
        # Noise 被过滤；只有 OK 被记录
        assert v.stable_label == "OK"

    def test_window_evicts_old_votes(self):
        v = WeightedVoter(window=3)
        v.update("A", 0.9)
        # 窗口装满后，更早的 A 会被挤出
        v.update("B", 0.6)
        v.update("B", 0.6)
        v.update("B", 0.6)
        assert v.stable_label == "B"

    def test_returns_label_immediately_first_call(self):
        v = WeightedVoter(window=5)
        out = v.update("OK", 0.9)
        assert out == "OK"

    def test_below_min_score_first_call_returns_input(self):
        """首次 update 即被过滤时，返回当前传入的 label（避免 None）。"""
        v = WeightedVoter(window=5, min_score=0.5)
        out = v.update("Noise", 0.1)
        assert out == "Noise"

    def test_stable_score_average_of_winning_label(self):
        v = WeightedVoter(window=5)
        v.update("OK", 0.8)
        v.update("OK", 1.0)
        v.update("Other", 0.5)
        # 胜出 OK 的平均权重 = (0.8 + 1.0) / 2 = 0.9
        assert v.stable_score == pytest.approx(0.9)

    def test_stable_score_empty_returns_zero(self):
        v = WeightedVoter(window=5)
        assert v.stable_score == 0.0

    def test_reset_clears_buffer(self):
        v = WeightedVoter(window=5)
        v.update("A", 0.9)
        v.reset()
        assert v.stable_label is None
        assert v.stable_score == 0.0

    def test_weighted_voter_recovers_faster_than_majority(self):
        """场景：5 帧序列，前 4 帧低置信 A，第 5 帧高置信 B。
        简单 MajorityVoter 会输出 A（4>1）；
        加权投票应该输出 B（如果 B 的高置信抵得过 A 的多次低置信）。"""
        # 我们设计 B 权重 > A 总权重
        wv = WeightedVoter(window=5)
        for _ in range(4):
            wv.update("A", 0.3)  # 4 * 0.3 = 1.2
        wv.update("B", 1.5)      # B 单帧 = 1.5 > 1.2
        assert wv.stable_label == "B"

        mv = MajorityVoter(window=5)
        for _ in range(4):
            mv.update("A")
        mv.update("B")
        assert mv.stable_label == "A"  # 多数投票仍输出 A
