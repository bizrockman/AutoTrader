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
from evolution.evaluator import evaluate
from evolution.generator import GeneratedStrategy, StrategyGenerator
from evolution.loop_detector import LoopDetector
from evolution.meta import MetaTracker
from knowledge.store import KnowledgeStore
from strategy.runner import StrategyRunner

log = logging.getLogger(__name__)


@dataclass
class ActiveStrategy:
    """A strategy currently running in the pool."""
    strategy_id: str
    run_id: int
    started_at: float
    eval_period_sec: int
    exchange: PaperExchange  # Each strategy gets its own paper exchange
    crash_count: int = 0


class Orchestrator:
    def __init__(self, config: Config):
        self.config = config
        self.store = KnowledgeStore(config.db_path)
        self.generator = StrategyGenerator(
            default_model=config.default_model,
            model_pool=config.model_pool,
        )
        self.runner = StrategyRunner()
        self.loop_detector = LoopDetector()
        self.meta = MetaTracker()
        self.connector = BinanceConnector(config.binance_api_key, config.binance_secret)

        self._active: dict[str, ActiveStrategy] = {}
        self._pending_results: list[dict] = []
        self._running = False
        self._current_wave_id: int | None = None

        # Price cache — fetch once per tick, share across all strategies
        self._last_price: float = 0.0
        self._last_price_time: float = 0.0

        # Candle tracking — detect when a new 1m candle closes
        self._last_candle_minute: int = -1

    async def start(self) -> None:
        """Start the orchestrator."""
        log.info("Starting AutoTrader Orchestrator")
        await self.store.connect()
        self._running = True

        try:
            # Initial generation
            await self._run_evolution_wave("initial")

            # Main loop
            while self._running:
                await self._tick_loop()
                await asyncio.sleep(5)  # Check every 5 seconds

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

    async def _capture_market_snapshot(self, symbol: str) -> dict:
        """Capture current market state for context."""
        try:
            price = await self.connector.get_price(symbol)
            ohlcv_1h = await self.connector.get_ohlcv(symbol, "1h", limit=24)
            ohlcv_1d = await self.connector.get_ohlcv(symbol, "1d", limit=7)

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
            regime = self._classify_regime(ohlcv_1d, price)

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
    def _classify_regime(ohlcv_daily: list[list], current_price: float) -> str:
        """Simple regime classification. No ML needed."""
        if len(ohlcv_daily) < 3:
            return "unknown"

        closes = [c[4] for c in ohlcv_daily]
        sma = sum(closes) / len(closes)

        # ATR (average true range) as volatility measure
        trs = []
        for i in range(1, len(ohlcv_daily)):
            high, low, prev_close = ohlcv_daily[i][2], ohlcv_daily[i][3], ohlcv_daily[i - 1][4]
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            trs.append(tr)
        atr = sum(trs) / len(trs) if trs else 0
        atr_pct = (atr / current_price * 100) if current_price > 0 else 0

        # Trend: is price consistently above/below SMA?
        above_sma = current_price > sma
        sma_slope = (closes[-1] - closes[0]) / len(closes) if len(closes) > 1 else 0
        sma_slope_pct = (sma_slope / sma * 100) if sma > 0 else 0

        if atr_pct > 5:  # High volatility
            if abs(sma_slope_pct) > 2:
                return "trending_up" if sma_slope_pct > 0 else "trending_down"
            return "volatile"
        elif above_sma and sma_slope_pct > 0.5:
            return "trending_up"
        elif not above_sma and sma_slope_pct < -0.5:
            return "trending_down"
        else:
            return "range"

    async def _tick_loop(self) -> None:
        """One iteration of the main loop: feed ticks, check harvests, maybe evolve."""
        symbol = self.config.default_symbol

        # 1. Fetch price ONCE, share across all strategies (with retry)
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

        # 2. Check if a new 1m candle just closed
        current_minute = int(timestamp // 60)
        new_candle = current_minute > self._last_candle_minute and self._last_candle_minute >= 0
        candle_data = None
        if new_candle:
            try:
                ohlcv = await self.connector.get_ohlcv(symbol, "1m", limit=2)
                if len(ohlcv) >= 2:
                    c = ohlcv[-2]  # The just-closed candle (not the current one)
                    candle_data = {
                        "timestamp": c[0], "open": c[1], "high": c[2],
                        "low": c[3], "close": c[4], "volume": c[5],
                    }
            except Exception as e:
                log.warning(f"Failed to fetch candle: {e}")
        self._last_candle_minute = current_minute

        # 3. Send tick (and candle if available) to all active strategies
        for strat_id, active in list(self._active.items()):
            try:
                # Always send tick
                result = await self.runner.send_tick(strat_id, symbol, price, timestamp, active.exchange)
                if result and result.get("type") == "signal" and result.get("result"):
                    await self._handle_signal(active, result["result"])
                elif result and result.get("type") == "error":
                    active.crash_count += 1

                # Send candle if one just closed
                if candle_data:
                    result = await self.runner.send_candle(strat_id, symbol, candle_data, timestamp, active.exchange)
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

        # Compute metrics
        eval_hours = active.eval_period_sec / 3600
        metrics = evaluate(
            trades=trades,
            initial_balance=self.config.initial_balance,
            final_balance=balance["total_value"],
            crash_count=active.crash_count,
            eval_period_hours=eval_hours,
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

        # Auto-promote to Hall of Fame if strategy performed well AND we're confident
        if metrics.pnl_pct > 0 and metrics.confidence >= 0.5 and metrics.fitness_score > 0.15:
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
        if health.waves_since_improvement >= 15:
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

        # Save to knowledge store
        await self.store.save_strategy(
            strategy_id=strategy_id,
            code=gen_strat.code,
            description=gen_strat.description,
            model_used=gen_strat.model_used,
            wave_id=wave_id,
        )

        # Track which blocks this strategy uses
        for block_name in (gen_strat.blocks_used or []):
            await self.store.record_block_usage(strategy_id, block_name)

        # Create a fresh paper exchange for this strategy
        paper_exchange = PaperExchange(
            connector=self.connector,
            initial_balance=self.config.initial_balance,
            fee_pct=self.config.trading_fee_pct,
        )

        # Use strategy-specific eval period (LLM decides based on timeframe)
        eval_period_sec = gen_strat.eval_period_minutes * 60

        # Capture market snapshot at deploy time
        start_snapshot = await self._capture_market_snapshot(self.config.default_symbol)

        # Start run in DB
        run_id = await self.store.start_run(strategy_id, eval_period_sec, start_snapshot=start_snapshot)

        # Start in Docker
        try:
            await self.runner.start_strategy(strategy_id, gen_strat.code, paper_exchange)
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
        )

        log.info(
            f"Deployed {strategy_id} — "
            f"eval: {gen_strat.eval_period_minutes}min — "
            f"model: {gen_strat.model_used} — "
            f"{gen_strat.description[:80]}"
        )
