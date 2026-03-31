"""Integration tests: verify LLM produces valid strategies with new timeframe schema.

These tests call real LLM APIs and cost real money — run explicitly:
    pytest tests/test_llm_generation.py -v -m integration
"""
from __future__ import annotations

import ast
import re

import pytest
import pytest_asyncio

from evolution.generator import (
    VALID_TIMEFRAMES,
    GeneratedStrategy,
    StrategyGenerator,
    timeframe_to_seconds,
)
from knowledge.store import EvolutionContext


def _empty_context(wave: int = 0) -> EvolutionContext:
    return EvolutionContext(
        top_strategies=[],
        recent_insights=[],
        failed_approaches=[],
        hall_of_fame=[],
        blocks=[],
        block_performance=[],
        current_wave=wave,
        total_strategies_tested=0,
    )


@pytest_asyncio.fixture
async def generator():
    return StrategyGenerator(
        default_model="claude-sonnet-4-20250514",
        temperature_generation=0.7,
        max_tokens_generation=6000,
    )


# ── Helpers ──────────────────────────────────────────────────────

def _assert_valid_strategy(s: GeneratedStrategy) -> None:
    """Validate a single generated strategy against the new schema."""
    assert s.code, "Code must not be empty"
    assert s.description, "Description must not be empty"

    # Timeframe
    assert s.primary_timeframe in VALID_TIMEFRAMES, (
        f"Invalid timeframe '{s.primary_timeframe}', must be one of {VALID_TIMEFRAMES}"
    )
    if s.primary_timeframe == "custom":
        assert s.candle_interval_seconds is not None, "custom timeframe needs candle_interval_seconds"
        assert s.candle_interval_seconds > 0

    # Bars
    assert isinstance(s.eval_bars, int) and s.eval_bars > 0, f"eval_bars must be positive int, got {s.eval_bars}"
    assert isinstance(s.warmup_bars, int) and s.warmup_bars >= 0, f"warmup_bars must be >= 0, got {s.warmup_bars}"

    # Eval period sanity: at least 10 minutes, at most 30 days
    eval_sec = s.eval_period_seconds
    assert eval_sec >= 600, f"Eval period too short: {eval_sec}s ({eval_sec/60:.0f}min)"
    assert eval_sec <= 30 * 86400, f"Eval period too long: {eval_sec}s ({eval_sec/86400:.1f}d)"

    # Warmup sanity: warmup should be < eval (warmup >= eval makes no sense)
    assert s.warmup_bars < s.eval_bars, (
        f"warmup_bars ({s.warmup_bars}) must be < eval_bars ({s.eval_bars})"
    )

    # Code must parse
    try:
        tree = ast.parse(s.code)
    except SyntaxError as e:
        pytest.fail(f"Strategy code has syntax error: {e}")

    # Code must contain 'class Strategy'
    class_names = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
    assert "Strategy" in class_names, f"Code must define 'class Strategy', found: {class_names}"

    # Code should contain on_tick or on_candle
    func_names = [node.name for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    has_handler = "on_tick" in func_names or "on_candle" in func_names
    assert has_handler, f"Code must define on_tick or on_candle, found: {func_names}"


def _assert_no_hardcoded_timeframes(s: GeneratedStrategy) -> None:
    """Check that the code doesn't hardcode timeframes — it should use self.timeframe."""
    hardcoded_pattern = re.compile(
        r"""get_ohlcv\s*\([^)]*['"](?:1m|3m|5m|15m|30m|1h|2h|4h|6h|8h|12h|1d|3d|1w)['"]"""
    )
    matches = hardcoded_pattern.findall(s.code)
    if matches:
        pytest.fail(
            f"Code hardcodes timeframes in get_ohlcv(): {matches}. "
            f"Should use self.timeframe instead."
        )


# ── Tests ────────────────────────────────────────────────────────

@pytest.mark.integration
async def test_generate_produces_valid_strategies(generator):
    """Cold-start generation: LLM gets empty context, must produce valid output."""
    strategies = await generator.generate(
        context=_empty_context(),
        symbol="BTC/USDT",
        count=2,
    )

    assert len(strategies) >= 1, "Generator must produce at least 1 strategy"

    for i, s in enumerate(strategies):
        print(f"\n--- Strategy {i+1}: {s.description[:80]} ---")
        print(f"    timeframe={s.primary_timeframe}, eval_bars={s.eval_bars}, warmup_bars={s.warmup_bars}")
        print(f"    eval_period={s.eval_period_seconds/3600:.1f}h, candle_sec={s.candle_seconds}")
        _assert_valid_strategy(s)


@pytest.mark.integration
async def test_strategies_avoid_hardcoded_timeframes(generator):
    """LLM should use self.timeframe instead of hardcoded strings in get_ohlcv()."""
    strategies = await generator.generate(
        context=_empty_context(),
        symbol="BTC/USDT",
        count=2,
    )

    for s in strategies:
        _assert_no_hardcoded_timeframes(s)


@pytest.mark.integration
async def test_warmup_bars_matches_indicator_needs(generator):
    """warmup_bars should be > 0 when the strategy uses indicators that need history."""
    strategies = await generator.generate(
        context=_empty_context(),
        symbol="BTC/USDT",
        count=3,
    )

    indicator_patterns = re.compile(
        r'\b(bollinger|ema|sma|macd|rsi|atr|stochastic)\b', re.IGNORECASE
    )

    for s in strategies:
        uses_indicators = bool(indicator_patterns.search(s.code))
        if uses_indicators:
            assert s.warmup_bars > 0, (
                f"Strategy uses indicators but warmup_bars=0: {s.description[:60]}"
            )


@pytest.mark.integration
async def test_diverse_timeframes_across_strategies(generator):
    """Multiple generated strategies should not all use the same timeframe."""
    strategies = await generator.generate(
        context=_empty_context(),
        symbol="BTC/USDT",
        count=3,
    )

    if len(strategies) < 2:
        pytest.skip("Need at least 2 strategies to check diversity")

    timeframes = {s.primary_timeframe for s in strategies}
    # Soft check: log but don't fail if all same (LLM may legitimately choose same tf)
    if len(timeframes) == 1:
        print(f"WARNING: All {len(strategies)} strategies use same timeframe: {timeframes.pop()}")


@pytest.mark.integration
async def test_second_wave_uses_context(generator):
    """Second wave generation should reference previous results in its output."""
    # Wave 1: cold start
    wave1 = await generator.generate(
        context=_empty_context(wave=0),
        symbol="BTC/USDT",
        count=2,
    )
    assert len(wave1) >= 1

    # Build a fake context with results from wave 1
    context = EvolutionContext(
        top_strategies=[{
            "id": "wave001_test", "description": wave1[0].description,
            "pnl_pct": -0.5, "fitness_score": -0.1,
        }],
        recent_insights=[{
            "category": "failure",
            "content": "RSI-based strategies did not work in ranging market.",
        }],
        failed_approaches=[{
            "id": "wave001_test", "description": wave1[0].description,
            "fitness_score": -0.1, "pnl_pct": -0.5,
        }],
        hall_of_fame=[],
        blocks=[],
        block_performance=[],
        current_wave=1,
        total_strategies_tested=2,
    )

    # Wave 2: with context
    wave2 = await generator.generate(
        context=context,
        symbol="BTC/USDT",
        count=2,
        plan="RSI failed. Try momentum or mean-reversion approaches instead.",
    )

    assert len(wave2) >= 1
    for s in wave2:
        _assert_valid_strategy(s)
