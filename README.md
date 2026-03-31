# AutoTrader

A self-evolving crypto trading bot. LLMs generate trading strategies as Python code, test them on live markets, and evolve them autonomously.

The system is simple. The complexity emerges.

## Concept

Inspired by [Karpathy's autoresearch](https://github.com/karpathy/autoresearch) and the [Meta-Harness paper](https://arxiv.org/abs/2603.28052): a simple loop + LLM + real feedback produces emergent complexity. Neither of those systems has loop detection, building block extraction, or statistical confidence scoring. This one does.

### The Loop

```
Generate strategies (LLM)  -->  Deploy on live markets (paper trading)
       ^                                |
       |                                v
  Plan + Blocks + Meta          Run with real market data
  from DB                       (each strategy decides its own timeframe)
       ^                                |
       |                                v
Analyze results (LLM)  <-----  Harvest when eval period ends
  - Extract building blocks             |
  - Write handover plan           Market snapshots captured
  - Update insights               at start + end of each run
  - Track evolution health
```

### Key Ideas

**Forward-testing only.** No backtesting. Strategies prove themselves on real market data. There's nothing to overfit on because the data doesn't exist yet when the strategy is created.

**Building blocks, not monolithic strategies.** The LLM extracts reusable components from successful strategies: indicators, signal generators, filters, risk management functions. These blocks are versioned and immutable. The system evolves on two levels simultaneously: inventing new blocks AND finding better combinations.

**Immutable history.** Nothing in the database is ever overwritten. Blocks get new versions, never updates. Every evolution wave saves a complete plan. You can cold-start from any wave or rerun the entire evolution from scratch.

**Two memory systems.** Within a session: conversation history gives the LLM short-term memory for making connections across waves. Between sessions: everything is in SQLite. A crash loses no data. The plan serves as a handover document for a fresh start.

**Statistical confidence.** Not all results are equal. 3 trades in 30 minutes is noise. 50 trades over 24 hours is signal. Every metric is weighted by a confidence score — uncertain results get pulled toward zero. The Hall of Fame only accepts strategies with confidence >= 0.5.

**Market context.** Every strategy run is tagged with market snapshots (start + end): price changes, volatility, and a regime classification (trending_up, trending_down, range, volatile). The LLM sees exactly what the market did during a run, so it can distinguish skill from luck.

**Meta-awareness.** The system monitors its own evolution: Is fitness trending up? Which models produce the best strategies? Is it stagnating? After 15 waves without improvement, it triggers a cold-start for a fresh perspective.

**Loop detection.** AST-based code similarity + ping-pong detection prevents the evolution from going in circles. Neither autoresearch nor Meta-Harness has this.

**Model diversity.** Uses LiteLLM to support multiple LLM providers. The model itself is a variable in the evolution: the system tracks which model produced which results.

## Architecture

```
AutoTrader/
+-- main.py                  # Entry point
+-- config.py                # Settings from .env
+-- Dockerfile               # Container for strategy execution
|
+-- exchange/
|   +-- connector.py         # Binance REST API via ccxt (price, OHLCV, orderbook,
|   |                        #   funding rate, open interest, trades, L/S ratio)
|   +-- paper.py             # Paper trading engine (real prices, virtual portfolio)
|
+-- strategy/
|   +-- template.py          # Strategy interface contract
|   +-- runner.py            # Docker-based execution + JSON-Lines proxy
|   +-- generated/           # Generated strategy files
|
+-- evolution/
|   +-- generator.py         # LLM briefings for generation + analysis
|   +-- evaluator.py         # Fitness metrics + confidence scoring
|   +-- loop_detector.py     # Duplicate detection + ping-pong detection
|   +-- meta.py              # Evolution health monitoring
|   +-- orchestrator.py      # Async main loop with overlapping generations
|
+-- knowledge/
    +-- store.py             # SQLite knowledge store (immutable append-only)
```

### Data Model

| Table | Purpose |
|---|---|
| `strategies` | Generated strategy code + metadata |
| `strategy_runs` | Execution records with market snapshots (start + end) |
| `trades` | Every trade with price, quantity, reason |
| `metrics` | Fitness scores + confidence per run |
| `blocks` | Versioned building blocks (immutable, never overwritten) |
| `strategy_blocks` | Which block versions a strategy used (its "genome") |
| `hall_of_fame` | Proven strategies (confidence >= 0.5) |
| `insights` | Learnings extracted by the analysis LLM |
| `evolution_waves` | Wave history with plans (handover documents) |
| `loop_checks` | Duplicate detection log |

### Strategy Interface

Strategies are Python classes with a simple contract:

```python
class Strategy:
    def __init__(self, exchange):
        self.exchange = exchange

    async def on_tick(self, symbol, price, timestamp) -> dict | None:
        # Called every ~5 seconds. Return None or a trade signal.

    async def on_candle(self, symbol, candle, timestamp) -> dict | None:
        # Called when a 1m candle closes. Same return format.

    def get_state(self) -> dict:
        # Internal state for logging.
```

### Available Market Data

The `exchange` object provides:

| Method | Data | Use Case |
|---|---|---|
| `get_price(symbol)` | Current price | Basic pricing |
| `get_ohlcv(symbol, timeframe, limit)` | Candles (any timeframe) | Technical analysis |
| `get_orderbook(symbol, limit)` | Bid/ask depth + spread | Liquidity, walls |
| `get_funding_rate(symbol)` | Funding rate, mark price | Sentiment (futures) |
| `get_open_interest(symbol)` | Total positioned | Crowding indicator |
| `get_recent_trades(symbol, limit)` | Trade flow with buy/sell side | Microstructure |
| `get_long_short_ratio(symbol)` | Account ratio | Retail positioning |
| `place_order(symbol, side, quantity)` | Execute trade | Paper or live |
| `get_balance()` | Portfolio state | Position sizing |
| `get_position(symbol)` | Current position | Risk management |

Strategies can use any combination. Not every strategy needs all data sources.

### Building Blocks

The most valuable output of the system. Blocks are versioned, immutable code components:

```
Block: adaptive_momentum v3 (indicator)
  depends_on: [vol_filter v2]
  used 12 times, avg fitness when used: 0.34, impact: +0.18
  origin: wave 12, strategy wave012_abc123

Block: regime_aware_entry v1 (composition)
  depends_on: [adaptive_momentum v3, vol_regime v2]
  used 5 times, avg fitness when used: 0.42, impact: +0.25
```

The LLM sees all blocks with their performance data and decides which to use, combine, or replace.

### Confidence Scoring

Not all results are equally trustworthy:

| Scenario | Confidence | Effect on Fitness |
|---|---|---|
| 3 trades, 30 min | 0.15 | Fitness * 0.15 (almost zeroed out) |
| 10 trades, 2 hours | 0.45 | Fitness halved |
| 20 trades, 6 hours | 0.65 | Fitness moderately weighted |
| 50 trades, 24 hours | 0.90 | Fitness nearly at face value |

Confidence is based on trade count (60% weight) and evaluation duration (40% weight). Strategies need confidence >= 0.5 to enter the Hall of Fame.

### Market Regime Classification

Every strategy run is tagged with the market regime at start and end:

| Regime | Conditions |
|---|---|
| `trending_up` | Price > SMA, SMA rising |
| `trending_down` | Price < SMA, SMA falling |
| `range` | Low ATR, price near SMA |
| `volatile` | High ATR, no clear direction |

This feeds into the analysis: blocks and strategies are tracked per regime.

### Evolution Plans

Every wave produces a plan, written by the analysis LLM as a handover document:

> "Momentum-based strategies outperformed in the last 3 waves (avg PnL +2.1%).
> The adaptive_momentum block combined with vol_filter shows consistent impact (+0.18).
> Next: try combining adaptive_momentum with mean_reversion_exit for range-bound markets.
> Avoid: pure RSI strategies have failed 4 times. The stochastic variants haven't been tested yet."

Plans are stored per wave. You can cold-start from any wave's plan.

### Meta-Tracking

The system monitors its own evolution:

- **Fitness trend**: Are strategies getting better over waves?
- **Model performance**: Which LLM produces the best strategies?
- **Stagnation detection**: Warning after 10 waves without improvement
- **Auto cold-start**: Fresh context after 15 waves of stagnation
- **Block diversity**: How many different blocks are in active use?

## Setup

```bash
# Clone and install
uv sync

# Configure
cp .env.example .env
# Edit .env with your API keys

# Build the strategy container
docker build -t autotrader-strategy .

# Run
uv run python main.py
```

### Requirements

- Python 3.11+
- Docker (for strategy isolation)
- Binance API key (for market data; no trading permissions needed for paper mode)
- LLM API key (Anthropic, OpenAI, or any LiteLLM-supported provider)

## Configuration

See [.env.example](.env.example):

| Variable | Description |
|---|---|
| `BINANCE_API_KEY` | Binance API key for market data |
| `DEFAULT_MODEL` | Default LLM for generation + analysis |
| `MODEL_POOL` | Comma-separated models for diversity |
| `INITIAL_BALANCE` | Paper trading start balance (USDT) |
| `TRADING_FEE_PCT` | Simulated fee per trade |
| `MAX_PARALLEL_STRATEGIES` | How many strategies run simultaneously |

## Status

MVP complete. Not yet tested end-to-end with live market data.

### Implemented
- Exchange connector (Binance via ccxt) + paper trading engine
- Extended market data: orderbook, funding rate, open interest, trade flow, L/S ratio
- Strategy runner with Docker isolation
- Evolution loop with overlapping generations
- LLM-based strategy generation and analysis (LiteLLM, multi-model)
- Building block extraction with immutable versioning
- Hall of Fame for proven strategies (confidence-gated)
- Confidence scoring on all metrics
- Market regime classification + snapshots per run
- Loop detection (AST similarity + ping-pong)
- Plan-based handover between waves
- Conversation history as short-term memory
- Meta-tracking: evolution health, model performance, stagnation detection
- Auto cold-start on prolonged stagnation

### Not Yet Implemented
- WebSocket market data (currently REST polling)
- State recovery after crash (paper exchange is in-memory)
- Multi-symbol support
- Real trading mode (live orders)
- LLM cost tracking per wave
- Dashboard / visualization
