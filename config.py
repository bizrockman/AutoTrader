from __future__ import annotations

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    # Binance
    binance_api_key: str = field(default_factory=lambda: os.getenv("BINANCE_API_KEY", ""))
    binance_secret: str = field(default_factory=lambda: os.getenv("BINANCE_SECRET", ""))

    # LLM
    default_model: str = field(default_factory=lambda: os.getenv("DEFAULT_MODEL", "claude-opus-4-6"))
    model_pool: list[str] = field(default_factory=lambda: os.getenv("MODEL_POOL", "claude-opus-4-6").split(","))

    # Paper Trading
    initial_balance: float = field(default_factory=lambda: float(os.getenv("INITIAL_BALANCE", "10000")))
    trading_fee_pct: float = field(default_factory=lambda: float(os.getenv("TRADING_FEE_PCT", "0.1")))
    default_symbol: str = field(default_factory=lambda: os.getenv("DEFAULT_SYMBOL", "BTC/USDT"))

    # Execution Reality Modeling
    slippage_pct: float = field(default_factory=lambda: float(os.getenv("SLIPPAGE_PCT", "0.01")))
    min_notional: float = field(default_factory=lambda: float(os.getenv("MIN_NOTIONAL", "5.0")))
    tick_size: float = field(default_factory=lambda: float(os.getenv("TICK_SIZE", "0.01")))
    step_size: float = field(default_factory=lambda: float(os.getenv("STEP_SIZE", "0.00001")))
    maker_fee_pct: float = field(default_factory=lambda: float(os.getenv("MAKER_FEE_PCT", "0.1")))
    taker_fee_pct: float = field(default_factory=lambda: float(os.getenv("TAKER_FEE_PCT", "0.1")))
    fill_delay_ms: int = field(default_factory=lambda: int(os.getenv("FILL_DELAY_MS", "0")))

    # Risk Governance (Track B — auskommentierte Defaults bis Implementierung)
    max_notional_per_strategy: float = field(default_factory=lambda: float(os.getenv("MAX_NOTIONAL_PER_STRATEGY", "0")))
    max_portfolio_exposure: float = field(default_factory=lambda: float(os.getenv("MAX_PORTFOLIO_EXPOSURE", "0")))
    max_daily_loss_pct: float = field(default_factory=lambda: float(os.getenv("MAX_DAILY_LOSS_PCT", "0")))
    max_drawdown_pct: float = field(default_factory=lambda: float(os.getenv("MAX_DRAWDOWN_PCT", "0")))
    max_consecutive_losses: int = field(default_factory=lambda: int(os.getenv("MAX_CONSECUTIVE_LOSSES", "0")))
    cooldown_minutes: int = field(default_factory=lambda: int(os.getenv("COOLDOWN_MINUTES", "0")))

    # Evolution
    max_parallel_strategies: int = field(default_factory=lambda: int(os.getenv("MAX_PARALLEL_STRATEGIES", "5")))

    # Paths
    db_path: str = field(default_factory=lambda: os.getenv("DB_PATH", "autotrader.db"))
    generated_strategies_dir: str = "strategy/generated"
