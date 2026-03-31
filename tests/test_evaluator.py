"""Tests for the evaluator — fitness scoring and confidence calculation.

Unit tests use synthetic trade data.
Integration tests (marked @pytest.mark.integration) feed real PaperExchange
trade history into the evaluator to verify the full pipeline.
"""
import pytest

from evolution.evaluator import evaluate


def _make_trade(side, price, qty, fee=0.0):
    return {"symbol": "BTC/USDT", "side": side, "price": price, "quantity": qty, "fee": fee}


# ── Unit tests ───────────────────────────────────────────────────────


def test_no_trades_zero_fitness():
    m = evaluate(trades=[], initial_balance=10_000, final_balance=10_000)
    assert m.trade_count == 0
    assert m.pnl == 0.0
    assert m.confidence < 0.1


def test_profitable_trades_positive_fitness():
    trades = [
        _make_trade("buy", 70_000, 0.01, fee=0.7),
        _make_trade("sell", 71_000, 0.01, fee=0.71),
    ]
    m = evaluate(trades=trades, initial_balance=10_000, final_balance=10_009, eval_period_hours=1)
    assert m.pnl > 0
    assert m.pnl_pct > 0
    assert m.win_rate > 0


def test_losing_trades_negative_pnl():
    trades = [
        _make_trade("buy", 70_000, 0.01, fee=0.7),
        _make_trade("sell", 69_000, 0.01, fee=0.69),
    ]
    m = evaluate(trades=trades, initial_balance=10_000, final_balance=9_989, eval_period_hours=1)
    assert m.pnl < 0
    assert m.pnl_pct < 0


def test_confidence_increases_with_trades():
    base_trades = [_make_trade("buy", 70_000, 0.01, 0.7), _make_trade("sell", 70_100, 0.01, 0.7)]

    few = evaluate(trades=base_trades * 1, initial_balance=10_000, final_balance=10_000, eval_period_hours=0.5)
    many = evaluate(trades=base_trades * 15, initial_balance=10_000, final_balance=10_000, eval_period_hours=6)

    assert many.confidence > few.confidence


def test_confidence_increases_with_duration():
    trades = [_make_trade("buy", 70_000, 0.01, 0.7), _make_trade("sell", 70_100, 0.01, 0.7)]

    short = evaluate(trades=trades, initial_balance=10_000, final_balance=10_000, eval_period_hours=0.5)
    long = evaluate(trades=trades, initial_balance=10_000, final_balance=10_000, eval_period_hours=24)

    assert long.confidence > short.confidence


def test_crash_count_penalizes_fitness():
    trades = [_make_trade("buy", 70_000, 0.01, 0.7), _make_trade("sell", 70_500, 0.01, 0.7)]

    clean = evaluate(trades=trades, initial_balance=10_000, final_balance=10_005, crash_count=0, eval_period_hours=2)
    crashy = evaluate(trades=trades, initial_balance=10_000, final_balance=10_005, crash_count=5, eval_period_hours=2)

    assert clean.fitness_score > crashy.fitness_score


def test_max_drawdown_computed():
    trades = [
        _make_trade("buy", 70_000, 0.1, fee=7.0),
        _make_trade("sell", 65_000, 0.1, fee=6.5),
    ]
    m = evaluate(trades=trades, initial_balance=10_000, final_balance=9_486.5, eval_period_hours=1)
    assert m.max_drawdown > 0


def test_metrics_to_dict():
    m = evaluate(trades=[], initial_balance=10_000, final_balance=10_000)
    d = m.to_dict()
    expected_keys = {"pnl", "pnl_pct", "sharpe", "max_drawdown", "trade_count", "win_rate", "crash_count", "fitness_score", "confidence"}
    assert expected_keys == set(d.keys())


# ── Integration tests (real Binance data → evaluator) ────────────────


@pytest.mark.integration
async def test_integration_evaluator_with_real_trades(real_exchange):
    """Execute a real buy/sell on PaperExchange, feed history into evaluator."""
    await real_exchange.place_order("BTC/USDT", "buy", 0.001)
    await real_exchange.place_order("BTC/USDT", "sell", 0.001)

    trades = real_exchange.get_trade_history()
    balance = await real_exchange.get_balance()

    m = evaluate(
        trades=trades,
        initial_balance=10_000.0,
        final_balance=balance["total_value"],
        eval_period_hours=0.01,
    )

    assert m.trade_count == 2
    assert m.win_rate is not None
    assert m.max_drawdown >= 0
    assert m.confidence > 0
    # Fees always eat into the balance, so PnL should be slightly negative
    # (buy+sell at ~same price minus 2x fee)
    assert m.pnl < 0
    assert m.fitness_score != 0.0  # should produce a non-trivial score


@pytest.mark.integration
async def test_integration_evaluator_metrics_are_serializable(real_exchange):
    """Metrics from real data can be serialized to dict (for DB storage)."""
    await real_exchange.place_order("BTC/USDT", "buy", 0.001)
    await real_exchange.place_order("BTC/USDT", "sell", 0.001)

    trades = real_exchange.get_trade_history()
    balance = await real_exchange.get_balance()

    m = evaluate(
        trades=trades,
        initial_balance=10_000.0,
        final_balance=balance["total_value"],
    )

    d = m.to_dict()
    assert isinstance(d, dict)
    assert all(isinstance(v, (int, float)) for v in d.values())
