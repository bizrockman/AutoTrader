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


@dataclass
class FitnessWeights:
    """Tunable weights for the composite fitness score."""
    pnl: float = 0.40
    sharpe: float = 0.25
    winrate: float = 0.15
    drawdown: float = 0.10
    activity: float = 0.05
    crash: float = 0.05


@dataclass
class ConfidenceParams:
    """Tunable parameters for the confidence curve."""
    trade_halflife: float = 15.0
    duration_halflife: float = 12.0


def evaluate(
    trades: list[dict],
    initial_balance: float,
    final_balance: float,
    crash_count: int = 0,
    eval_period_hours: float = 1.0,
    warmup_bars: int = 0,
    candle_seconds: int = 300,
    fitness_weights: FitnessWeights | None = None,
    confidence_params: ConfidenceParams | None = None,
) -> StrategyMetrics:
    """Compute fitness metrics for a strategy run.

    Trades that occurred during the warmup phase are excluded from evaluation.
    """
    fw = fitness_weights or FitnessWeights()
    cp = confidence_params or ConfidenceParams()

    warmup_cutoff = warmup_bars * candle_seconds if warmup_bars > 0 else 0

    if warmup_cutoff > 0 and trades:
        first_ts = _trade_timestamp(trades[0])
        trades = [t for t in trades if _trade_timestamp(t) - first_ts >= warmup_cutoff]

    pnl = final_balance - initial_balance
    pnl_pct = (pnl / initial_balance * 100) if initial_balance > 0 else 0.0

    trade_count = len(trades)
    confidence = _compute_confidence(trade_count, eval_period_hours, cp)

    if trade_count == 0:
        raw_fitness = _compute_fitness(pnl_pct, 0.0, 0.0, 0, 0.0, crash_count, fw)
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

    trade_pnls = _compute_trade_pnls(trades)
    wins = sum(1 for p in trade_pnls if p > 0)
    win_rate = (wins / len(trade_pnls) * 100) if trade_pnls else 0.0

    sharpe = _compute_sharpe(trade_pnls, eval_period_hours)
    max_drawdown = _compute_max_drawdown(trades, initial_balance)

    raw_fitness = _compute_fitness(pnl_pct, sharpe, max_drawdown, trade_count, win_rate, crash_count, fw)
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


def _compute_confidence(trade_count: int, eval_period_hours: float, params: ConfidenceParams | None = None) -> float:
    """How much to trust this result. 0.0 = noise, 1.0 = statistically meaningful."""
    p = params or ConfidenceParams()
    trade_factor = 1 - math.exp(-trade_count / p.trade_halflife) if trade_count > 0 else 0
    duration_factor = 1 - math.exp(-eval_period_hours / p.duration_halflife)
    confidence = 0.6 * trade_factor + 0.4 * duration_factor
    return round(min(confidence, 1.0), 3)


def _trade_timestamp(trade: dict) -> float:
    """Extract a numeric timestamp from a trade dict (epoch seconds)."""
    ts = trade.get("timestamp", 0)
    if isinstance(ts, str):
        from datetime import datetime, timezone
        try:
            return datetime.fromisoformat(ts).replace(tzinfo=timezone.utc).timestamp()
        except ValueError:
            return 0.0
    return float(ts)


def _compute_fitness(
    pnl_pct: float,
    sharpe: float,
    max_drawdown: float,
    trade_count: int,
    win_rate: float,
    crash_count: int,
    weights: FitnessWeights | None = None,
) -> float:
    """Composite fitness score. Higher is better."""
    w = weights or FitnessWeights()

    pnl_score = max(min(pnl_pct / 5.0, 2.0), -2.0)
    sharpe_score = max(min(sharpe / 2.0, 2.0), -2.0)
    wr_score = (win_rate - 50) / 50
    dd_penalty = -min(max_drawdown / 10.0, 2.0)

    if trade_count == 0:
        activity_score = -1.0
    elif trade_count <= 50:
        activity_score = min(trade_count / 10.0, 1.0)
    else:
        activity_score = max(1.0 - (trade_count - 50) / 100.0, -0.5)

    crash_penalty = -crash_count * 0.5

    fitness = (
        w.pnl * pnl_score
        + w.sharpe * sharpe_score
        + w.winrate * wr_score
        + w.drawdown * dd_penalty
        + w.activity * activity_score
        + w.crash * crash_penalty
    )

    return round(fitness, 4)
