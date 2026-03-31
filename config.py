from __future__ import annotations

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


def _float(key: str, default: str) -> float:
    return float(os.getenv(key, default))


def _int(key: str, default: str) -> int:
    return int(os.getenv(key, default))


@dataclass
class Config:
    # ── Binance ──────────────────────────────────────────────────
    binance_api_key: str = field(default_factory=lambda: os.getenv("BINANCE_API_KEY", ""))
    binance_secret: str = field(default_factory=lambda: os.getenv("BINANCE_SECRET", ""))
    api_timeout_sec: int = field(default_factory=lambda: _int("API_TIMEOUT_SEC", "15"))

    # ── LLM ──────────────────────────────────────────────────────
    default_model: str = field(default_factory=lambda: os.getenv("DEFAULT_MODEL", "claude-opus-4-6"))
    model_pool: list[str] = field(default_factory=lambda: os.getenv("MODEL_POOL", "claude-opus-4-6").split(","))
    llm_temperature_generation: float = field(default_factory=lambda: _float("LLM_TEMPERATURE_GENERATION", "0.8"))
    llm_temperature_analysis: float = field(default_factory=lambda: _float("LLM_TEMPERATURE_ANALYSIS", "0.3"))
    llm_max_tokens_generation: int = field(default_factory=lambda: _int("LLM_MAX_TOKENS_GENERATION", "8000"))
    llm_max_tokens_analysis: int = field(default_factory=lambda: _int("LLM_MAX_TOKENS_ANALYSIS", "6000"))
    llm_default_model_ratio: float = field(default_factory=lambda: _float("LLM_DEFAULT_MODEL_RATIO", "0.7"))
    llm_history_max_turns: int = field(default_factory=lambda: _int("LLM_HISTORY_MAX_TURNS", "10"))

    # ── Paper Trading ────────────────────────────────────────────
    initial_balance: float = field(default_factory=lambda: _float("INITIAL_BALANCE", "10000"))
    trading_fee_pct: float = field(default_factory=lambda: _float("TRADING_FEE_PCT", "0.1"))
    default_symbol: str = field(default_factory=lambda: os.getenv("DEFAULT_SYMBOL", "BTC/USDT"))

    # ── Execution Reality Modeling ───────────────────────────────
    slippage_pct: float = field(default_factory=lambda: _float("SLIPPAGE_PCT", "0.01"))
    min_notional: float = field(default_factory=lambda: _float("MIN_NOTIONAL", "5.0"))
    tick_size: float = field(default_factory=lambda: _float("TICK_SIZE", "0.01"))
    step_size: float = field(default_factory=lambda: _float("STEP_SIZE", "0.00001"))
    maker_fee_pct: float = field(default_factory=lambda: _float("MAKER_FEE_PCT", "0.1"))
    taker_fee_pct: float = field(default_factory=lambda: _float("TAKER_FEE_PCT", "0.1"))
    fill_delay_ms: int = field(default_factory=lambda: _int("FILL_DELAY_MS", "0"))

    # ── Risk Governance (0 = disabled) ───────────────────────────
    max_notional_per_strategy: float = field(default_factory=lambda: _float("MAX_NOTIONAL_PER_STRATEGY", "0"))
    max_portfolio_exposure: float = field(default_factory=lambda: _float("MAX_PORTFOLIO_EXPOSURE", "0"))
    max_daily_loss_pct: float = field(default_factory=lambda: _float("MAX_DAILY_LOSS_PCT", "0"))
    max_drawdown_pct: float = field(default_factory=lambda: _float("MAX_DRAWDOWN_PCT", "0"))
    max_consecutive_losses: int = field(default_factory=lambda: _int("MAX_CONSECUTIVE_LOSSES", "0"))
    cooldown_minutes: int = field(default_factory=lambda: _int("COOLDOWN_MINUTES", "0"))

    # ── Evolution ────────────────────────────────────────────────
    max_parallel_strategies: int = field(default_factory=lambda: _int("MAX_PARALLEL_STRATEGIES", "5"))
    tick_interval_sec: int = field(default_factory=lambda: _int("TICK_INTERVAL_SEC", "5"))
    stagnation_wave_threshold: int = field(default_factory=lambda: _int("STAGNATION_WAVE_THRESHOLD", "15"))

    # ── Docker / Runner ──────────────────────────────────────────
    docker_memory: str = field(default_factory=lambda: os.getenv("DOCKER_MEMORY", "256m"))
    docker_cpus: str = field(default_factory=lambda: os.getenv("DOCKER_CPUS", "0.5"))
    docker_ready_timeout_sec: int = field(default_factory=lambda: _int("DOCKER_READY_TIMEOUT_SEC", "30"))
    strategy_tick_timeout_sec: int = field(default_factory=lambda: _int("STRATEGY_TICK_TIMEOUT_SEC", "10"))

    # ── Fitness / Evaluator ──────────────────────────────────────
    fitness_weight_pnl: float = field(default_factory=lambda: _float("FITNESS_WEIGHT_PNL", "0.40"))
    fitness_weight_sharpe: float = field(default_factory=lambda: _float("FITNESS_WEIGHT_SHARPE", "0.25"))
    fitness_weight_winrate: float = field(default_factory=lambda: _float("FITNESS_WEIGHT_WINRATE", "0.15"))
    fitness_weight_drawdown: float = field(default_factory=lambda: _float("FITNESS_WEIGHT_DRAWDOWN", "0.10"))
    fitness_weight_activity: float = field(default_factory=lambda: _float("FITNESS_WEIGHT_ACTIVITY", "0.05"))
    fitness_weight_crash: float = field(default_factory=lambda: _float("FITNESS_WEIGHT_CRASH", "0.05"))
    confidence_trade_halflife: float = field(default_factory=lambda: _float("CONFIDENCE_TRADE_HALFLIFE", "15"))
    confidence_duration_halflife: float = field(default_factory=lambda: _float("CONFIDENCE_DURATION_HALFLIFE", "12"))

    # ── Hall of Fame Thresholds ──────────────────────────────────
    hof_min_pnl_pct: float = field(default_factory=lambda: _float("HOF_MIN_PNL_PCT", "0"))
    hof_min_confidence: float = field(default_factory=lambda: _float("HOF_MIN_CONFIDENCE", "0.5"))
    hof_min_fitness: float = field(default_factory=lambda: _float("HOF_MIN_FITNESS", "0.15"))

    # ── Market Snapshot / Regime ─────────────────────────────────
    snapshot_hourly_limit: int = field(default_factory=lambda: _int("SNAPSHOT_HOURLY_LIMIT", "24"))
    snapshot_daily_limit: int = field(default_factory=lambda: _int("SNAPSHOT_DAILY_LIMIT", "7"))
    regime_volatility_pct: float = field(default_factory=lambda: _float("REGIME_VOLATILITY_PCT", "5.0"))
    regime_trend_pct: float = field(default_factory=lambda: _float("REGIME_TREND_PCT", "2.0"))
    regime_slope_pct: float = field(default_factory=lambda: _float("REGIME_SLOPE_PCT", "0.5"))

    # ── Paths ────────────────────────────────────────────────────
    db_path: str = field(default_factory=lambda: os.getenv("DB_PATH", "autotrader.db"))
    generated_strategies_dir: str = field(default_factory=lambda: os.getenv("GENERATED_STRATEGIES_DIR", "strategy/generated"))

    @property
    def quote_currency(self) -> str:
        """Derive quote currency from symbol (e.g. 'BTC/USDT' -> 'USDT')."""
        return self.default_symbol.split("/")[-1] if "/" in self.default_symbol else "USDT"
