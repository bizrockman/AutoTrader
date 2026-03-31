"""Tests for timeframe normalization and candle aggregation."""
from __future__ import annotations

import pytest

from evolution.generator import (
    VALID_TIMEFRAMES,
    GeneratedStrategy,
    timeframe_to_seconds,
)
from evolution.orchestrator import CandleAccumulator
from evolution.evaluator import evaluate


# ── timeframe_to_seconds ────────────────────────────────────────

class TestTimeframeToSeconds:
    def test_standard_timeframes(self):
        assert timeframe_to_seconds("1m") == 60
        assert timeframe_to_seconds("5m") == 300
        assert timeframe_to_seconds("1h") == 3600
        assert timeframe_to_seconds("1d") == 86400

    def test_tick_returns_poll_interval(self):
        assert timeframe_to_seconds("tick") == 5

    def test_custom_uses_provided_seconds(self):
        assert timeframe_to_seconds("custom", 450) == 450
        assert timeframe_to_seconds("custom", 90) == 90

    def test_custom_without_seconds_falls_back(self):
        assert timeframe_to_seconds("custom", None) == 60

    def test_unknown_falls_back_to_5m(self):
        assert timeframe_to_seconds("banana") == 300


# ── GeneratedStrategy properties ────────────────────────────────

class TestGeneratedStrategyProperties:
    def test_candle_seconds_standard(self):
        s = GeneratedStrategy(code="", description="", model_used="test", primary_timeframe="15m")
        assert s.candle_seconds == 900

    def test_candle_seconds_custom(self):
        s = GeneratedStrategy(
            code="", description="", model_used="test",
            primary_timeframe="custom", candle_interval_seconds=450,
        )
        assert s.candle_seconds == 450

    def test_eval_period_seconds(self):
        s = GeneratedStrategy(
            code="", description="", model_used="test",
            primary_timeframe="5m", eval_bars=300,
        )
        assert s.eval_period_seconds == 300 * 300  # 300 bars * 300s = 90000s = 25h

    def test_warmup_seconds(self):
        s = GeneratedStrategy(
            code="", description="", model_used="test",
            primary_timeframe="1h", warmup_bars=20,
        )
        assert s.warmup_seconds == 20 * 3600

    def test_tick_eval_period(self):
        s = GeneratedStrategy(
            code="", description="", model_used="test",
            primary_timeframe="tick", eval_bars=720,
        )
        assert s.eval_period_seconds == 720 * 5  # 3600s = 1h


# ── CandleAccumulator ───────────────────────────────────────────

class TestCandleAccumulator:
    def test_first_tick_returns_none(self):
        acc = CandleAccumulator(interval_sec=60)
        assert acc.update(100.0, 1000.0) is None

    def test_ticks_within_interval_return_none(self):
        acc = CandleAccumulator(interval_sec=60)
        acc.update(100.0, 960.0)  # aligns to 960
        assert acc.update(101.0, 965.0) is None
        assert acc.update(99.0, 970.0) is None

    def test_boundary_crossing_returns_candle(self):
        acc = CandleAccumulator(interval_sec=60)
        acc.update(100.0, 960.0)
        acc.update(105.0, 980.0)
        acc.update(98.0, 1000.0)

        candle = acc.update(102.0, 1020.0)  # crosses 960+60=1020
        assert candle is not None
        assert candle["open"] == 100.0
        assert candle["high"] == 105.0
        assert candle["low"] == 98.0
        assert candle["close"] == 98.0

    def test_custom_interval(self):
        acc = CandleAccumulator(interval_sec=450)  # 7.5 min
        acc.update(100.0, 0.0)
        assert acc.update(101.0, 200.0) is None
        candle = acc.update(99.0, 450.0)
        assert candle is not None
        assert candle["open"] == 100.0


# ── Evaluator warmup filtering ──────────────────────────────────

class TestEvaluatorWarmup:
    def _make_trades(self, timestamps: list[float]) -> list[dict]:
        trades = []
        for i, ts in enumerate(timestamps):
            side = "buy" if i % 2 == 0 else "sell"
            trades.append({
                "symbol": "BTC/USDT", "side": side,
                "quantity": 0.001, "price": 50000.0, "fee": 0.5,
                "timestamp": ts,
            })
        return trades

    def test_warmup_zero_keeps_all_trades(self):
        trades = self._make_trades([100, 200, 400, 500])
        m = evaluate(trades, 10000, 10010, warmup_bars=0, candle_seconds=60)
        assert m.trade_count == 4

    def test_warmup_filters_early_trades(self):
        # warmup = 5 bars * 60s = 300s cutoff
        trades = self._make_trades([100, 200, 500, 600])
        m = evaluate(trades, 10000, 10010, warmup_bars=5, candle_seconds=60)
        # first trade at 100, cutoff at 100+300=400 -> trades at 500,600 survive
        assert m.trade_count == 2

    def test_warmup_all_filtered_gives_zero_trades(self):
        trades = self._make_trades([10, 20, 30, 40])
        m = evaluate(trades, 10000, 10000, warmup_bars=100, candle_seconds=60)
        assert m.trade_count == 0


# ── VALID_TIMEFRAMES ────────────────────────────────────────────

class TestValidTimeframes:
    def test_contains_standard(self):
        for tf in ["1m", "5m", "15m", "1h", "4h", "1d"]:
            assert tf in VALID_TIMEFRAMES

    def test_contains_custom(self):
        assert "custom" in VALID_TIMEFRAMES

    def test_contains_tick(self):
        assert "tick" in VALID_TIMEFRAMES
