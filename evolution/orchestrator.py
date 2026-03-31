"""Orchestrator — the main evolution loop.

Manages overlapping generations of strategies:
- Spawns new strategies when slots are free
- Harvests results when strategies reach their eval period
- Triggers evolution when enough results are collected
- Runs 24/7
"""
from __future__ import annotations

import asyncio
import logging
import math
import time
import uuid
from dataclasses import dataclass, field

from config import Config
from exchange.connector import BinanceConnector
from exchange.paper import PaperExchange
from evolution.evaluator import ConfidenceParams, FitnessWeights, evaluate
from evolution.generator import (
    BINANCE_TIMEFRAMES,
    GeneratedStrategy,
    StrategyGenerator,
    timeframe_to_seconds,
)
from evolution.loop_detector import LoopDetector
from evolution.meta import MetaTracker
from knowledge.store import KnowledgeStore
from strategy.runner import StrategyRunner

log = logging.getLogger(__name__)


@dataclass
class CandleAccumulator:
    """Builds a candle from ticks for one timeframe interval."""
    interval_sec: int
    current_start: float = 0.0
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0
    volume: float = 0.0
    tick_count: int = 0

    def update(self, price: float, ts: float) -> dict | None:
        """Feed a tick. Returns completed candle dict if interval boundary crossed, else None."""
        if self.tick_count == 0:
            self.current_start = (ts // self.interval_sec) * self.interval_sec
            self.open = self.high = self.low = self.close = price
            self.tick_count = 1
            return None

        candle_end = self.current_start + self.interval_sec
        if ts >= candle_end:
            completed = {
                "timestamp": int(self.current_start * 1000),
                "open": self.open, "high": self.high,
                "low": self.low, "close": self.close,
                "volume": self.volume,
            }
            self.current_start = (ts // self.interval_sec) * self.interval_sec
            self.open = self.high = self.low = self.close = price
            self.volume = 0.0
            self.tick_count = 1
            return completed

        self.high = max(self.high, price)
        self.low = min(self.low, price)
        self.close = price
        self.tick_count += 1
        return None


@dataclass
class ActiveStrategy:
    """A strategy currently running in the pool."""
    strategy_id: str
    run_id: int
    started_at: float
    eval_period_sec: int
    exchange: PaperExchange
    primary_timeframe: str = "5m"
    candle_seconds: int = 300
    warmup_bars: int = 0
    crash_count: int = 0
    accumulator: CandleAccumulator | None = None


class Orchestrator:
    def __init__(self, config: Config):
        self.config = config
        self.store = KnowledgeStore(config.db_path)
        self.generator = StrategyGenerator(
            default_model=config.default_model,
            model_pool=config.model_pool,
            temperature_generation=config.llm_temperature_generation,
            temperature_analysis=config.llm_temperature_analysis,
            max_tokens_generation=config.llm_max_tokens_generation,
            max_tokens_analysis=config.llm_max_tokens_analysis,
            default_model_ratio=config.llm_default_model_ratio,
            history_max_turns=config.llm_history_max_turns,
            fee_pct=config.trading_fee_pct,
            tick_interval_sec=config.tick_interval_sec,
        )
        self.runner = StrategyRunner(
            docker_memory=config.docker_memory,
            docker_cpus=config.docker_cpus,
            ready_timeout_sec=config.docker_ready_timeout_sec,
            tick_timeout_sec=config.strategy_tick_timeout_sec,
        )
        self.loop_detector = LoopDetector()
        self.meta = MetaTracker()
        self.connector = BinanceConnector(
            config.binance_api_key, config.binance_secret, timeout_sec=config.api_timeout_sec,
        )

        self._fitness_weights = FitnessWeights(
            pnl=config.fitness_weight_pnl,
            sharpe=config.fitness_weight_sharpe,
            winrate=config.fitness_weight_winrate,
            drawdown=config.fitness_weight_drawdown,
            activity=config.fitness_weight_activity,
            crash=config.fitness_weight_crash,
        )
        self._confidence_params = ConfidenceParams(
            trade_halflife=config.confidence_trade_halflife,
            duration_halflife=config.confidence_duration_halflife,
        )

        self._active: dict[str, ActiveStrategy] = {}
        self._pending_results: list[dict] = []
        self._running = False
        self._current_wave_id: int | None = None

        self._last_price: float = 0.0
        self._last_price_time: float = 0.0

        # Per-timeframe candle cache: avoid fetching the same Binance candle N times
        self._candle_cache: dict[str, tuple[float, dict]] = {}  # tf -> (fetched_ts, candle)
        self._last_candle_boundary: dict[str, int] = {}  # tf -> last boundary epoch

    async def start(self) -> None:
        """Start the orchestrator."""
        log.info("Starting AutoTrader Orchestrator")
        await self.store.connect()
        self._running = True

        try:
            # Clean up stale runs from previous process
            interrupted = await self.store.interrupt_stale_runs()
            if interrupted:
                log.info(f"Marked {interrupted} stale run(s) as interrupted")

            # Resume existing strategies before generating new ones
            resumed = await self._resume_strategies()

            # Only generate new strategies if we still have empty slots
            free_slots = self.config.max_parallel_strategies - len(self._active)
            if free_slots > 0:
                log.info(f"{free_slots} slot(s) free after resume — generating new strategies")
                await self._run_evolution_wave("initial" if resumed == 0 else "fill_slots")
            else:
                log.info(f"All {len(self._active)} slot(s) filled from resume — skipping generation")

            # Main loop
            while self._running:
                await self._tick_loop()
                await asyncio.sleep(self.config.tick_interval_sec)

        except KeyboardInterrupt:
            log.info("Shutting down...")
        finally:
            await self.shutdown()

    async def shutdown(self) -> None:
        """Graceful shutdown."""
        self._running = False
        log.info("Stopping all strategies...")
        await self.runner.stop_all()
        await self.connector.close()
        await self.store.close()
        log.info("Orchestrator stopped.")

    async def _resume_strategies(self) -> int:
        """Redeploy strategies from the DB instead of generating new ones."""
        free_slots = self.config.max_parallel_strategies - len(self._active)
        if free_slots <= 0:
            return 0

        resumable = await self.store.get_resumable_strategies(limit=free_slots)
        if not resumable:
            log.info("No existing strategies to resume — will generate fresh")
            return 0

        log.info(f"Resuming {len(resumable)} strategy/ies from DB")
        deployed = 0

        for strat in resumable:
            if len(self._active) >= self.config.max_parallel_strategies:
                break

            strategy_id = strat["strategy_id"]
            code = strat["code"]
            tf = strat.get("primary_timeframe", "5m")
            eval_bars = strat.get("eval_bars", 300)
            warmup = strat.get("warmup_bars", 0)
            custom_sec = strat.get("candle_interval_seconds")
            candle_sec = timeframe_to_seconds(tf, custom_sec)
            eval_period_sec = eval_bars * candle_sec

            paper_exchange = PaperExchange(
                connector=self.connector,
                initial_balance=self.config.initial_balance,
                fee_pct=self.config.trading_fee_pct,
                quote_currency=self.config.quote_currency,
            )

            start_snapshot = await self._capture_market_snapshot(self.config.default_symbol)
            run_id = await self.store.start_run(strategy_id, eval_period_sec, start_snapshot=start_snapshot)

            try:
                await self.runner.start_strategy(strategy_id, code, paper_exchange, timeframe=tf)
            except Exception as e:
                log.error(f"Failed to resume strategy {strategy_id}: {e}")
                await self.store.finish_run(run_id, status="crashed", error_message=str(e))
                continue

            self._active[strategy_id] = ActiveStrategy(
                strategy_id=strategy_id,
                run_id=run_id,
                started_at=time.time(),
                eval_period_sec=eval_period_sec,
                exchange=paper_exchange,
                primary_timeframe=tf,
                candle_seconds=candle_sec,
                warmup_bars=warmup,
            )

            eval_min = round(eval_period_sec / 60)
            desc = strat.get("description", "")[:80]
            log.info(f"Resumed {strategy_id} — tf: {tf}, eval: {eval_bars} bars (~{eval_min}min) — {desc}")
            deployed += 1

        return deployed

    async def _capture_market_snapshot(self, symbol: str) -> dict:
        """Capture current market state for context."""
        try:
            price = await self.connector.get_price(symbol)
            ohlcv_1h = await self.connector.get_ohlcv(symbol, "1h", limit=self.config.snapshot_hourly_limit)
            ohlcv_1d = await self.connector.get_ohlcv(symbol, "1d", limit=self.config.snapshot_daily_limit)

            # Calculate changes from OHLCV
            change_1h = ((price - ohlcv_1h[-2][4]) / ohlcv_1h[-2][4] * 100) if len(ohlcv_1h) >= 2 else 0
            change_24h = ((price - ohlcv_1h[0][1]) / ohlcv_1h[0][1] * 100) if ohlcv_1h else 0
            change_7d = ((price - ohlcv_1d[0][1]) / ohlcv_1d[0][1] * 100) if ohlcv_1d else 0

            # 24h volatility: std of hourly returns
            if len(ohlcv_1h) >= 2:
                hourly_returns = [
                    (ohlcv_1h[i][4] - ohlcv_1h[i - 1][4]) / ohlcv_1h[i - 1][4]
                    for i in range(1, len(ohlcv_1h))
                    if ohlcv_1h[i - 1][4] > 0
                ]
                vol_24h = (sum(r ** 2 for r in hourly_returns) / len(hourly_returns)) ** 0.5 * 100 if hourly_returns else 0
            else:
                vol_24h = 0

            # Regime classification
            regime = self._classify_regime(
                ohlcv_1d, price,
                self.config.regime_volatility_pct,
                self.config.regime_trend_pct,
                self.config.regime_slope_pct,
            )

            return {
                "price": price,
                "change_1h_pct": round(change_1h, 2),
                "change_24h_pct": round(change_24h, 2),
                "change_7d_pct": round(change_7d, 2),
                "volatility_24h_pct": round(vol_24h, 2),
                "regime": regime,
                "timestamp": time.time(),
            }
        except Exception as e:
            log.warning(f"Failed to capture market snapshot: {e}")
            return {"price": self._last_price, "regime": "unknown", "timestamp": time.time()}

    @staticmethod
    def _classify_regime(
        ohlcv_daily: list[list],
        current_price: float,
        vol_threshold: float = 5.0,
        trend_threshold: float = 2.0,
        slope_threshold: float = 0.5,
    ) -> str:
        """Simple regime classification."""
        if len(ohlcv_daily) < 3:
            return "unknown"

        closes = [c[4] for c in ohlcv_daily]
        sma = sum(closes) / len(closes)

        trs = []
        for i in range(1, len(ohlcv_daily)):
            high, low, prev_close = ohlcv_daily[i][2], ohlcv_daily[i][3], ohlcv_daily[i - 1][4]
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            trs.append(tr)
        atr = sum(trs) / len(trs) if trs else 0
        atr_pct = (atr / current_price * 100) if current_price > 0 else 0

        above_sma = current_price > sma
        sma_slope = (closes[-1] - closes[0]) / len(closes) if len(closes) > 1 else 0
        sma_slope_pct = (sma_slope / sma * 100) if sma > 0 else 0

        if atr_pct > vol_threshold:
            if abs(sma_slope_pct) > trend_threshold:
                return "trending_up" if sma_slope_pct > 0 else "trending_down"
            return "volatile"
        elif above_sma and sma_slope_pct > slope_threshold:
            return "trending_up"
        elif not above_sma and sma_slope_pct < -slope_threshold:
            return "trending_down"
        else:
            return "range"

    async def _tick_loop(self) -> None:
        """One iteration of the main loop: feed ticks, check harvests, maybe evolve."""
        symbol = self.config.default_symbol

        # 1. Fetch price ONCE
        price = None
        for attempt in range(3):
            try:
                price = await self.connector.get_price(symbol)
                self._last_price = price
                self._last_price_time = time.time()
                break
            except Exception as e:
                if attempt == 2:
                    if self._last_price > 0:
                        log.warning(f"Price fetch failed 3x, using last known: {self._last_price}")
                        price = self._last_price
                    else:
                        log.error(f"Price fetch failed, no fallback: {e}")
                        return
                await asyncio.sleep(2 ** attempt)

        timestamp = time.time()

        # 2. Per-strategy: send tick + timeframe-aware candle dispatch
        for strat_id, active in list(self._active.items()):
            try:
                # Always send tick
                result = await self.runner.send_tick(strat_id, symbol, price, timestamp, active.exchange)
                if result and result.get("type") == "signal" and result.get("result"):
                    await self._handle_signal(active, result["result"])
                elif result and result.get("type") == "error":
                    active.crash_count += 1

                # Skip candle dispatch for tick-only strategies
                if active.primary_timeframe == "tick":
                    continue

                candle = await self._get_candle_for(active, symbol, price, timestamp)
                if candle:
                    result = await self.runner.send_candle(strat_id, symbol, candle, timestamp, active.exchange)
                    if result and result.get("type") == "signal" and result.get("result"):
                        await self._handle_signal(active, result["result"])
                    elif result and result.get("type") == "error":
                        active.crash_count += 1

            except Exception as e:
                active.crash_count += 1
                log.error(f"Error with strategy {strat_id}: {e}")

        # 3. Harvest strategies that reached their eval period
        now = time.time()
        to_harvest = [
            sid for sid, a in self._active.items()
            if now - a.started_at >= a.eval_period_sec
        ]
        for sid in to_harvest:
            await self._harvest_strategy(sid)

        # 4. Trigger evolution when there are results AND free slots
        free_slots = self.config.max_parallel_strategies - len(self._active)
        if self._pending_results and free_slots > 0:
            await self._run_evolution_wave(
                f"harvested_{len(self._pending_results)}_strategies"
            )

    async def _get_candle_for(self, active: ActiveStrategy, symbol: str, price: float, ts: float) -> dict | None:
        """Get a just-closed candle for this strategy's timeframe, or None."""
        tf = active.primary_timeframe
        sec = active.candle_seconds

        # Standard Binance timeframes: fetch from API (better volume/OHLC data)
        if tf in BINANCE_TIMEFRAMES:
            boundary = int(ts // sec)
            last = self._last_candle_boundary.get(tf, -1)
            if boundary <= last:
                return None
            self._last_candle_boundary[tf] = boundary

            # Check cache to avoid duplicate API calls for same timeframe
            cached = self._candle_cache.get(tf)
            if cached and cached[0] == boundary:
                return cached[1]

            try:
                ohlcv = await self.connector.get_ohlcv(symbol, tf, limit=2)
                if len(ohlcv) >= 2:
                    c = ohlcv[-2]
                    candle = {
                        "timestamp": c[0], "open": c[1], "high": c[2],
                        "low": c[3], "close": c[4], "volume": c[5],
                    }
                    self._candle_cache[tf] = (boundary, candle)
                    return candle
            except Exception as e:
                log.warning(f"Failed to fetch {tf} candle: {e}")
            return None

        # Custom timeframes: build from tick accumulator
        if active.accumulator is None:
            active.accumulator = CandleAccumulator(interval_sec=sec)
        return active.accumulator.update(price, ts)

    async def _handle_signal(self, active: ActiveStrategy, signal: dict) -> None:
        """Process a trade signal from a strategy."""
        if signal.get("action") in ("buy", "sell"):
            result = await active.exchange.place_order(
                signal["symbol"], signal["action"], signal["quantity"],
            )
            if result["status"] == "filled":
                await self.store.record_trade(
                    strategy_id=active.strategy_id,
                    run_id=active.run_id,
                    symbol=signal["symbol"],
                    side=signal["action"],
                    quantity=signal["quantity"],
                    price=result["price"],
                    fee=result["fee"],
                    reason=signal.get("reason"),
                )
                log.info(
                    f"[{active.strategy_id}] {signal['action'].upper()} "
                    f"{signal['quantity']} {signal['symbol']} @ {result['price']:.2f} "
                    f"— {signal.get('reason', 'no reason')}"
                )

    async def _harvest_strategy(self, strategy_id: str) -> None:
        """Harvest a completed strategy: compute metrics, store results."""
        active = self._active.pop(strategy_id, None)
        if not active:
            return

        log.info(f"Harvesting strategy {strategy_id}")

        # Stop the container
        await self.runner.stop_strategy(strategy_id)

        # Get trade history and balance
        trades = active.exchange.get_trade_history()
        balance = await active.exchange.get_balance()

        eval_hours = active.eval_period_sec / 3600
        metrics = evaluate(
            trades=trades,
            initial_balance=self.config.initial_balance,
            final_balance=balance["total_value"],
            crash_count=active.crash_count,
            eval_period_hours=eval_hours,
            warmup_bars=active.warmup_bars,
            candle_seconds=active.candle_seconds,
            fitness_weights=self._fitness_weights,
            confidence_params=self._confidence_params,
        )

        # Capture end-of-run market snapshot
        end_snapshot = await self._capture_market_snapshot(self.config.default_symbol)

        # Store
        await self.store.save_metrics(active.run_id, strategy_id, metrics.to_dict())
        await self.store.finish_run(active.run_id, status="completed", end_snapshot=end_snapshot)

        # Get strategy description + start snapshot for results
        strat = await self.store.get_strategy(strategy_id)
        description = strat["description"] if strat else ""

        # Build market context for the analysis LLM
        market_context = f"Regime: {end_snapshot.get('regime', '?')}"
        start_snap = None
        async with self.store.db.execute(
            "SELECT start_snapshot FROM strategy_runs WHERE id = ?", (active.run_id,)
        ) as cur:
            row = await cur.fetchone()
            if row and row[0]:
                import json
                start_snap = json.loads(row[0])
        if start_snap:
            price_change = end_snapshot.get('price', 0) - start_snap.get('price', 0)
            price_change_pct = (price_change / start_snap['price'] * 100) if start_snap.get('price') else 0
            market_context = (
                f"Market during run: {start_snap.get('regime', '?')} → {end_snapshot.get('regime', '?')}, "
                f"BTC moved {price_change_pct:+.2f}%, "
                f"24h vol: {end_snapshot.get('volatility_24h_pct', '?')}%"
            )

        self._pending_results.append({
            "strategy_id": strategy_id,
            "description": description,
            "code": strat["code"] if strat else "",
            "market_context": market_context,
            **metrics.to_dict(),
            "sample_trades": trades[:10],
        })

        log.info(
            f"Strategy {strategy_id} completed: PnL={metrics.pnl:.2f} ({metrics.pnl_pct:.2f}%), "
            f"Fitness={metrics.fitness_score:.4f}, Trades={metrics.trade_count}"
        )

        if (metrics.pnl_pct > self.config.hof_min_pnl_pct
                and metrics.confidence >= self.config.hof_min_confidence
                and metrics.fitness_score > self.config.hof_min_fitness):
            await self.store.promote_to_hall_of_fame(
                strategy_id=strategy_id,
                summary=description,
                best_pnl_pct=metrics.pnl_pct,
                best_sharpe=metrics.sharpe,
                avg_fitness=metrics.fitness_score,
            )
            log.info(f"*** {strategy_id} promoted to Hall of Fame (fitness: {metrics.fitness_score:.4f}) ***")

    async def _run_evolution_wave(self, trigger: str) -> None:
        """Run a full evolution wave: analyze → generate → deploy."""
        log.info(f"=== Evolution Wave (trigger: {trigger}) ===")

        wave_id = await self.store.start_wave(trigger)
        self._current_wave_id = wave_id

        # 1. Analyze previous results (if any) — fresh LLM call
        plan = await self.store.get_latest_plan()
        analysis = None

        if self._pending_results:
            existing_blocks = await self.store.get_blocks()
            analysis = await self.generator.analyze(
                symbol=self.config.default_symbol,
                strategy_results=self._pending_results,
                existing_blocks=existing_blocks,
            )
            log.info(f"Analysis: {analysis.get('analysis', '')[:200]}")

            # Save insights
            for insight in analysis.get("insights", []):
                await self.store.save_insight(
                    wave_id=wave_id,
                    category=insight.get("category", "meta"),
                    content=insight.get("content", ""),
                    confidence=insight.get("confidence", 0.5),
                    model_used=self.config.default_model,
                )

            # Extract reusable building blocks
            for block in analysis.get("extracted_blocks", []):
                if block.get("name") and block.get("code"):
                    await self.store.save_block(
                        name=block["name"],
                        code=block["code"],
                        description=block.get("description", ""),
                        category=block.get("category", "utility"),
                        depends_on=block.get("depends_on"),
                        origin_strategy=block.get("origin_strategy"),
                        origin_wave=wave_id,
                    )
                    log.info(f"Extracted block: {block['name']} ({block.get('category', 'utility')})")

            # Save the plan for the next wave — this is the handover document
            new_plan = analysis.get("plan", "")
            if new_plan:
                await self.store.save_wave_plan(wave_id, new_plan)
                plan = new_plan
                log.info(f"Plan for next wave: {new_plan[:150]}")

            # Save wave results
            await self.store.finish_wave(
                wave_id=wave_id,
                strategies_evaluated=[r["strategy_id"] for r in self._pending_results],
                analysis=analysis.get("analysis", ""),
                decisions=analysis,
                model_used=self.config.default_model,
            )

            self._pending_results.clear()

        # 2. Meta-health check
        health = await self.meta.get_health(self.store)
        log.info(f"Evolution health: trend={health.fitness_trend:+.3f}, "
                 f"best={health.best_fitness_ever:.3f}, stagnating={health.is_stagnating}")

        # Auto cold-start if stagnating hard
        if health.waves_since_improvement >= self.config.stagnation_wave_threshold:
            log.warning("Severe stagnation — triggering cold start for fresh perspective")
            self.generator.cold_start()

        # 3. Build loop detection context
        all_strategies = await self.store.get_all_strategy_summaries()
        exploration_summary = self.loop_detector.get_exploration_summary(all_strategies)
        ping_pong = self.loop_detector.detect_ping_pong(all_strategies)
        loop_warning = self.loop_detector.build_loop_warning(None, ping_pong, exploration_summary)

        # Add meta-health to loop warning so the LLM sees it
        loop_warning = health.to_prompt_section() + "\n\n" + loop_warning

        # 4. Generate new strategies — fresh LLM call with full DB briefing
        context = await self.store.get_evolution_context()
        free_slots = self.config.max_parallel_strategies - len(self._active)
        count = max(free_slots, 1)

        # Capture current market snapshot for the generation prompt
        current_snapshot = await self._capture_market_snapshot(self.config.default_symbol)
        market_info = (
            f"\n## Current Market\n"
            f"Price: ${current_snapshot.get('price', 0):,.2f}, "
            f"Regime: {current_snapshot.get('regime', 'unknown')}, "
            f"24h change: {current_snapshot.get('change_24h_pct', 0):+.1f}%, "
            f"Volatility: {current_snapshot.get('volatility_24h_pct', 0):.1f}%"
        )
        if plan:
            plan = market_info + "\n\n" + plan
        else:
            plan = market_info

        generated = await self.generator.generate(
            context=context,
            symbol=self.config.default_symbol,
            count=count,
            loop_warning=loop_warning,
            plan=plan,
        )

        # 4. Validate and deploy
        for gen_strat in generated:
            # Loop detection check
            check = self.loop_detector.check_similarity(
                gen_strat.code, gen_strat.description, all_strategies,
            )
            if check.is_duplicate:
                log.warning(f"Loop detected: {check.recommendation}")
                await self.store.save_loop_check(
                    strategy_id="pending",
                    similarity_score=check.similarity_score,
                    most_similar_id=check.most_similar_id,
                    action_taken="rejected",
                )
                continue

            # Deploy
            if len(self._active) >= self.config.max_parallel_strategies:
                log.info("All slots full, skipping remaining strategies")
                break

            await self._deploy_strategy(gen_strat, wave_id)

    async def _deploy_strategy(self, gen_strat: GeneratedStrategy, wave_id: int) -> None:
        """Deploy a generated strategy."""
        strategy_id = f"wave{wave_id:03d}_{uuid.uuid4().hex[:8]}"

        await self.store.save_strategy(
            strategy_id=strategy_id,
            code=gen_strat.code,
            description=gen_strat.description,
            model_used=gen_strat.model_used,
            wave_id=wave_id,
            primary_timeframe=gen_strat.primary_timeframe,
            eval_bars=gen_strat.eval_bars,
            warmup_bars=gen_strat.warmup_bars,
            candle_interval_seconds=gen_strat.candle_interval_seconds,
        )

        for block_name in (gen_strat.blocks_used or []):
            await self.store.record_block_usage(strategy_id, block_name)

        paper_exchange = PaperExchange(
            connector=self.connector,
            initial_balance=self.config.initial_balance,
            fee_pct=self.config.trading_fee_pct,
            quote_currency=self.config.quote_currency,
        )

        eval_period_sec = gen_strat.eval_period_seconds
        candle_sec = gen_strat.candle_seconds
        start_snapshot = await self._capture_market_snapshot(self.config.default_symbol)
        run_id = await self.store.start_run(strategy_id, eval_period_sec, start_snapshot=start_snapshot)

        try:
            await self.runner.start_strategy(
                strategy_id, gen_strat.code, paper_exchange,
                timeframe=gen_strat.primary_timeframe,
            )
        except Exception as e:
            log.error(f"Failed to start strategy {strategy_id}: {e}")
            await self.store.finish_run(run_id, status="crashed", error_message=str(e))
            return

        self._active[strategy_id] = ActiveStrategy(
            strategy_id=strategy_id,
            run_id=run_id,
            started_at=time.time(),
            eval_period_sec=eval_period_sec,
            exchange=paper_exchange,
            primary_timeframe=gen_strat.primary_timeframe,
            candle_seconds=candle_sec,
            warmup_bars=gen_strat.warmup_bars,
        )

        eval_minutes = round(eval_period_sec / 60)
        log.info(
            f"Deployed {strategy_id} — "
            f"tf: {gen_strat.primary_timeframe}, eval: {gen_strat.eval_bars} bars (~{eval_minutes}min), "
            f"warmup: {gen_strat.warmup_bars} bars — "
            f"model: {gen_strat.model_used} — "
            f"{gen_strat.description[:80]}"
        )
