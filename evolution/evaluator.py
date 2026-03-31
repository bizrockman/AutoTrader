"""Strategy Evaluator — computes fitness metrics from trade results."""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class StrategyMetrics:
    pnl: float
    pnl_pct: float
    sharpe: float
    max_drawdown: float
    trade_count: int
    win_rate: float
    crash_count: int
    fitness_score: float
    confidence: float  # 0.0-1.0: how much to trust this result

    def to_dict(self) -> dict:
        return {
            "pnl": self.pnl,
            "pnl_pct": self.pnl_pct,
            "sharpe": self.sharpe,
            "max_drawdown": self.max_drawdown,
            "trade_count": self.trade_count,
            "win_rate": self.win_rate,
            "crash_count": self.crash_count,
            "fitness_score": self.fitness_score,
            "confidence": self.confidence,
        }


def evaluate(
    trades: list[dict],
    initial_balance: float,
    final_balance: float,
    crash_count: int = 0,
    eval_period_hours: float = 1.0,
) -> StrategyMetrics:
    """Compute fitness metrics for a strategy run."""
    pnl = final_balance - initial_balance
    pnl_pct = (pnl / initial_balance * 100) if initial_balance > 0 else 0.0

    # Trade analysis
    trade_count = len(trades)
    confidence = _compute_confidence(trade_count, eval_period_hours)

    if trade_count == 0:
        raw_fitness = _compute_fitness(pnl_pct, 0.0, 0.0, 0, 0.0, crash_count)
        return StrategyMetrics(
            pnl=pnl,
            pnl_pct=pnl_pct,
            sharpe=0.0,
            max_drawdown=0.0,
            trade_count=0,
            win_rate=0.0,
            crash_count=crash_count,
            fitness_score=raw_fitness * confidence,
            confidence=confidence,
        )

    # Compute per-trade PnL (pair buys with sells)
    trade_pnls = _compute_trade_pnls(trades)
    wins = sum(1 for p in trade_pnls if p > 0)
    win_rate = (wins / len(trade_pnls) * 100) if trade_pnls else 0.0

    # Sharpe-like ratio (annualized from the evaluation period)
    sharpe = _compute_sharpe(trade_pnls, eval_period_hours)

    # Max drawdown from equity curve
    max_drawdown = _compute_max_drawdown(trades, initial_balance)

    raw_fitness = _compute_fitness(pnl_pct, sharpe, max_drawdown, trade_count, win_rate, crash_count)

    # Confidence-weighted fitness: uncertain results get pulled toward 0
    fitness = raw_fitness * confidence

    return StrategyMetrics(
        pnl=pnl,
        pnl_pct=pnl_pct,
        sharpe=sharpe,
        max_drawdown=max_drawdown,
        trade_count=trade_count,
        win_rate=win_rate,
        crash_count=crash_count,
        fitness_score=fitness,
        confidence=confidence,
    )


def _compute_trade_pnls(trades: list[dict]) -> list[float]:
    """Compute PnL for each round-trip (buy→sell pair)."""
    pnls = []
    open_buys: dict[str, list[dict]] = {}

    for t in trades:
        symbol = t["symbol"]
        if t["side"] == "buy":
            open_buys.setdefault(symbol, []).append(t)
        elif t["side"] == "sell" and open_buys.get(symbol):
            buy = open_buys[symbol].pop(0)
            buy_cost = buy["price"] * buy["quantity"] + buy.get("fee", 0)
            sell_revenue = t["price"] * t["quantity"] - t.get("fee", 0)
            pnls.append(sell_revenue - buy_cost)

    return pnls


def _compute_sharpe(trade_pnls: list[float], eval_period_hours: float) -> float:
    """Compute a Sharpe-like ratio."""
    if len(trade_pnls) < 2:
        return 0.0

    mean = sum(trade_pnls) / len(trade_pnls)
    variance = sum((p - mean) ** 2 for p in trade_pnls) / (len(trade_pnls) - 1)
    std = math.sqrt(variance) if variance > 0 else 0.001

    # Annualize: scale by sqrt of trades-per-year estimate
    trades_per_hour = len(trade_pnls) / max(eval_period_hours, 0.01)
    trades_per_year = trades_per_hour * 24 * 365
    annualization = math.sqrt(max(trades_per_year, 1))

    return (mean / std) * annualization


def _compute_max_drawdown(trades: list[dict], initial_balance: float) -> float:
    """Compute max drawdown percentage from trade sequence."""
    equity = initial_balance
    peak = equity
    max_dd = 0.0

    for t in trades:
        cost = t["price"] * t["quantity"]
        fee = t.get("fee", 0)
        if t["side"] == "buy":
            equity -= cost + fee
        else:
            equity += cost - fee

        peak = max(peak, equity)
        dd = (peak - equity) / peak * 100 if peak > 0 else 0
        max_dd = max(max_dd, dd)

    return max_dd


def _compute_confidence(trade_count: int, eval_period_hours: float) -> float:
    """How much to trust this result. 0.0 = noise, 1.0 = statistically meaningful.

    Three factors:
    1. Trade count: need ~20+ trades for any statistical significance
    2. Eval duration: longer = seen more market conditions
    3. Combined with diminishing returns (sigmoid-like)
    """
    # Trade count factor: 0 trades = 0, 20 trades = ~0.7, 50+ = ~0.95
    trade_factor = 1 - math.exp(-trade_count / 15) if trade_count > 0 else 0

    # Duration factor: 1h = 0.3, 6h = 0.7, 24h = 0.9, 72h+ = ~1.0
    duration_factor = 1 - math.exp(-eval_period_hours / 12)

    # Combine: both matter, but trade count matters more
    confidence = 0.6 * trade_factor + 0.4 * duration_factor

    return round(min(confidence, 1.0), 3)


def _compute_fitness(
    pnl_pct: float,
    sharpe: float,
    max_drawdown: float,
    trade_count: int,
    win_rate: float,
    crash_count: int,
) -> float:
    """Composite fitness score. Higher is better.

    Weighted combination:
    - PnL% (40%): primary objective
    - Sharpe (25%): risk-adjusted return
    - Win Rate (15%): consistency
    - Drawdown penalty (10%): capital preservation
    - Activity bonus/penalty (5%): reward some trading, penalize inactivity
    - Crash penalty (5%): stability
    """
    # Normalize components to roughly [-1, 1] range
    pnl_score = max(min(pnl_pct / 5.0, 2.0), -2.0)  # ±5% maps to ±1
    sharpe_score = max(min(sharpe / 2.0, 2.0), -2.0)  # ±2 maps to ±1
    wr_score = (win_rate - 50) / 50  # 50% → 0, 100% → 1, 0% → -1
    dd_penalty = -min(max_drawdown / 10.0, 2.0)  # -10% DD → -1

    # Activity: reward 5-50 trades, penalize 0 or excessive
    if trade_count == 0:
        activity_score = -1.0
    elif trade_count <= 50:
        activity_score = min(trade_count / 10.0, 1.0)
    else:
        activity_score = max(1.0 - (trade_count - 50) / 100.0, -0.5)

    crash_penalty = -crash_count * 0.5

    fitness = (
        0.40 * pnl_score
        + 0.25 * sharpe_score
        + 0.15 * wr_score
        + 0.10 * dd_penalty
        + 0.05 * activity_score
        + 0.05 * crash_penalty
    )

    return round(fitness, 4)
