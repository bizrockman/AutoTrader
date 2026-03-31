"""Strategy Generator — uses LLMs to create and evolve trading strategies.

Architecture:
- Within a session: maintain conversation history (short-term memory)
- Between sessions: everything is saved to DB (plans, blocks, insights)
- Cold-start: load latest plan + all artifacts from DB → full recovery
- Every plan is a snapshot: complete history of reasoning, rerunnable
"""
from __future__ import annotations

import json
import logging
import random
import re
from dataclasses import dataclass, field

import litellm

from knowledge.store import EvolutionContext
from strategy.template import build_interface_doc

log = logging.getLogger(__name__)


TIMEFRAME_SECONDS: dict[str, int] = {
    "tick": 0,
    "1m": 60, "3m": 180, "5m": 300, "15m": 900, "30m": 1800,
    "1h": 3600, "2h": 7200, "4h": 14400,
    "6h": 21600, "8h": 28800, "12h": 43200,
    "1d": 86400, "3d": 259200, "1w": 604800,
}

BINANCE_TIMEFRAMES = set(TIMEFRAME_SECONDS.keys()) - {"tick"}

VALID_TIMEFRAMES = set(TIMEFRAME_SECONDS.keys()) | {"custom"}


def timeframe_to_seconds(tf: str, custom_seconds: int | None = None) -> int:
    """Convert a timeframe string to seconds. Returns 5 for 'tick' (poll interval)."""
    if tf == "tick":
        return 5
    if tf == "custom":
        return custom_seconds or 60
    return TIMEFRAME_SECONDS.get(tf, 300)


@dataclass
class GeneratedStrategy:
    code: str
    description: str
    model_used: str
    primary_timeframe: str = "5m"
    eval_bars: int = 300
    warmup_bars: int = 0
    candle_interval_seconds: int | None = None
    blocks_used: list[str] | None = None

    @property
    def candle_seconds(self) -> int:
        return timeframe_to_seconds(self.primary_timeframe, self.candle_interval_seconds)

    @property
    def eval_period_seconds(self) -> int:
        return self.eval_bars * self.candle_seconds

    @property
    def warmup_seconds(self) -> int:
        return self.warmup_bars * self.candle_seconds


_SYSTEM_PROMPT_TEMPLATE = """You are an autonomous trading strategy researcher.
You write Python trading strategies, deploy them on live crypto markets,
observe the results, and iterate. Your goal: find strategies that make money.

{interface_doc}

Write COMPLETE, RUNNABLE Python code. Use `await` for all exchange calls.

You may be given a conversation history from earlier waves. Use it —
those connections and intuitions are valuable. But don't rely on it:
everything important is also in the briefing data (blocks, metrics, plans).
If the conversation starts fresh, you lose nothing critical.
"""


GENERATION_PROMPT = """## Briefing: Wave {wave}
- Strategies tested so far: {total_tested}
- Symbol: {symbol}

{plan_section}

## What Has Worked
{top_performers}

## What Has Failed
{failed}

## Learnings
{insights}

## Hall of Fame
{hall_of_fame}

## Building Blocks
{blocks}

## Block Impact Rankings
{block_impact}

{loop_warning}

## Task
Write {count} new strategies.

For each strategy, you MUST choose the execution parameters:

- **primary_timeframe**: The candle interval your strategy operates on.
  Valid: "tick", "1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "custom"
  Use "tick" for microstructure strategies (orderbook, trade flow).
  Use "custom" + candle_interval_seconds for non-standard intervals (e.g. 450 = 7.5 min).
  The engine will call on_candle() at your chosen interval. on_tick() is always called every ~5s.

- **eval_bars**: How many bars (candles) to observe before evaluating. Think in bars, not minutes.
  Example: 300 bars of 5m = 25 hours. 300 bars of 1h = 12.5 days.
  For "tick" mode, 1 bar = 1 tick (~5 seconds). 720 bars = 1 hour.
  More bars = higher confidence but longer evaluation. 200-500 is typical.

- **warmup_bars**: How many bars your indicators need before generating valid signals.
  Example: 20-period Bollinger Band needs warmup_bars >= 20.
  Trades during warmup are ignored by the evaluator.

```json
[
  {{
    "description": "What this does and why.",
    "primary_timeframe": "5m",
    "eval_bars": 300,
    "warmup_bars": 30,
    "blocks_used": ["block_name"],
    "code": "class Strategy:\\n    ..."
  }}
]
```
"""


ANALYSIS_PROMPT = """## Wave Results

Symbol: {symbol}

{strategy_results}

## Existing Blocks
{existing_blocks}

## Task
Analyze results. Extract reusable blocks. Write a plan for the next wave.

The plan is your most important output — it's your handover document.
Write it so that you (or another model) can pick up exactly where you left off.
Include: what you tried, what you learned, what to try next, which block
combinations look promising, what's still unexplored.

```json
{{
  "analysis": "Specific analysis with numbers.",
  "insights": [
    {{"category": "pattern|failure|market_regime|technique", "content": "...", "confidence": 0.0-1.0}}
  ],
  "next_directions": ["..."],
  "extracted_blocks": [
    {{
      "name": "block_name",
      "category": "indicator|signal|filter|risk|utility|composition",
      "description": "What it does, when it works, when it doesn't.",
      "code": "def block_name(...):\\n    ...",
      "depends_on": ["other_block"],
      "origin_strategy": "strategy_id"
    }}
  ],
  "plan": "Detailed research plan. What to focus on, what to avoid, promising combinations, open questions."
}}
```
"""


class StrategyGenerator:
    """Manages LLM calls for strategy generation and analysis.

    Maintains conversation history within a session for short-term memory.
    All artifacts are saved to DB independently — a cold-start loses no data.
    """

    def __init__(
        self,
        default_model: str = "claude-opus-4-6",
        model_pool: list[str] | None = None,
        temperature_generation: float = 0.8,
        temperature_analysis: float = 0.3,
        max_tokens_generation: int = 8000,
        max_tokens_analysis: int = 6000,
        default_model_ratio: float = 0.7,
        history_max_turns: int = 10,
        fee_pct: float = 0.1,
        tick_interval_sec: int = 5,
    ):
        self.default_model = default_model
        self.model_pool = model_pool or [default_model]
        self.temperature_generation = temperature_generation
        self.temperature_analysis = temperature_analysis
        self.max_tokens_generation = max_tokens_generation
        self.max_tokens_analysis = max_tokens_analysis
        self.default_model_ratio = default_model_ratio
        self.history_max_turns = history_max_turns

        self._system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(
            interface_doc=build_interface_doc(fee_pct=fee_pct, tick_interval_sec=tick_interval_sec)
        )

        self._generation_history: list[dict] = []
        self._analysis_history: list[dict] = []

    def _pick_model(self) -> str:
        if random.random() < self.default_model_ratio or len(self.model_pool) == 1:
            return self.default_model
        return random.choice(self.model_pool)

    async def generate(
        self,
        context: EvolutionContext,
        symbol: str,
        count: int = 3,
        loop_warning: str = "",
        plan: str = "",
    ) -> list[GeneratedStrategy]:
        """Generate strategies. Uses conversation history if available, DB briefing always."""
        model = self._pick_model()
        log.info(f"Generating {count} strategies with {model}")

        plan_section = ""
        if plan:
            plan_section = f"## Research Plan (from previous analysis)\n{plan}"

        prompt = GENERATION_PROMPT.format(
            wave=context.current_wave + 1,
            total_tested=context.total_strategies_tested,
            symbol=symbol,
            plan_section=plan_section,
            top_performers=self._fmt_strategies(context.top_strategies),
            failed=self._fmt_strategies(context.failed_approaches),
            insights=self._fmt_insights(context.recent_insights),
            hall_of_fame=self._fmt_hall_of_fame(context.hall_of_fame),
            blocks=self._fmt_blocks(context.blocks),
            block_impact=self._fmt_block_impact(context.block_performance),
            loop_warning=f"## Loop Detection\n{loop_warning}" if loop_warning else "",
            count=count,
        )

        messages = [{"role": "system", "content": self._system_prompt}]
        messages.extend(self._generation_history)
        messages.append({"role": "user", "content": prompt})

        response = await litellm.acompletion(
            model=model,
            messages=messages,
            temperature=self.temperature_generation,
            max_tokens=self.max_tokens_generation,
        )

        content = response.choices[0].message.content

        self._generation_history.append({"role": "user", "content": prompt})
        self._generation_history.append({"role": "assistant", "content": content})
        self._trim_history(self._generation_history, max_turns=self.history_max_turns)

        strategies = self._parse_strategies(content, model)
        log.info(f"Generated {len(strategies)} strategies")
        return strategies

    async def analyze(
        self,
        symbol: str,
        strategy_results: list[dict],
        existing_blocks: list[dict],
    ) -> dict:
        """Analyze results. Maintains analysis conversation history."""
        model = self.default_model
        log.info(f"Analyzing {len(strategy_results)} results with {model}")

        results_str = ""
        for r in strategy_results:
            results_str += f"\n### {r['strategy_id']}\n"
            results_str += f"Description: {r.get('description', 'N/A')}\n"
            if r.get("market_context"):
                results_str += f"**{r['market_context']}**\n"
            results_str += f"PnL: {r.get('pnl', 0):.2f} ({r.get('pnl_pct', 0):.2f}%)\n"
            results_str += f"Trades: {r.get('trade_count', 0)}, Win Rate: {r.get('win_rate', 0):.1f}%\n"
            results_str += f"Max Drawdown: {r.get('max_drawdown', 0):.2f}%\n"
            if r.get("sample_trades"):
                results_str += f"Sample trades: {json.dumps(r['sample_trades'][:5])}\n"
            if r.get("code"):
                results_str += f"Code:\n```python\n{r['code']}\n```\n"

        prompt = ANALYSIS_PROMPT.format(
            symbol=symbol,
            strategy_results=results_str,
            existing_blocks=self._fmt_blocks(existing_blocks),
        )

        messages = [
            {"role": "system", "content": "You are a quantitative trading analyst and research director."},
        ]
        messages.extend(self._analysis_history)
        messages.append({"role": "user", "content": prompt})

        response = await litellm.acompletion(
            model=model,
            messages=messages,
            temperature=self.temperature_analysis,
            max_tokens=self.max_tokens_analysis,
        )

        content = response.choices[0].message.content

        self._analysis_history.append({"role": "user", "content": prompt})
        self._analysis_history.append({"role": "assistant", "content": content})
        self._trim_history(self._analysis_history, max_turns=self.history_max_turns)

        return self._parse_analysis(content)

    def cold_start(self) -> None:
        """Clear conversation history. Next calls start fresh from DB only."""
        self._generation_history.clear()
        self._analysis_history.clear()
        log.info("Cold start: conversation history cleared. DB artifacts preserved.")

    @staticmethod
    def _trim_history(history: list[dict], max_turns: int = 10) -> None:
        """Keep only the last N turns (pairs of user+assistant messages)."""
        max_messages = max_turns * 2
        while len(history) > max_messages:
            history.pop(0)

    # ── Formatters ──────────────────────────────────────────────

    def _fmt_strategies(self, strategies: list[dict]) -> str:
        if not strategies:
            return "None yet."
        parts = []
        for s in strategies:
            parts.append(
                f"- **{s.get('id', '?')}**: {s.get('description', 'N/A')} "
                f"→ PnL: {s.get('pnl_pct', '?')}%, Fitness: {s.get('fitness_score', '?')}"
            )
        return "\n".join(parts)

    def _fmt_insights(self, insights: list[dict]) -> str:
        if not insights:
            return "No learnings yet."
        return "\n".join(f"- {i.get('content', '')}" for i in insights)

    def _fmt_hall_of_fame(self, hof: list[dict]) -> str:
        if not hof:
            return "No proven strategies yet."
        parts = []
        for s in hof:
            parts.append(
                f"\n### {s.get('strategy_id', '?')} — {s.get('summary', s.get('description', 'N/A'))}"
            )
            parts.append(
                f"Best PnL: {s.get('best_pnl_pct', '?')}%, "
                f"Runs: {s.get('total_runs', 1)}, "
                f"Avg Fitness: {s.get('avg_fitness', '?')}"
            )
            if s.get("code"):
                parts.append(f"```python\n{s['code']}\n```")
        return "\n".join(parts)

    def _fmt_blocks(self, blocks: list[dict]) -> str:
        if not blocks:
            return "No building blocks yet — start from scratch."
        parts = []
        for b in blocks:
            used = b.get('avg_fitness_when_used')
            perf = f" — avg fitness: {used:.3f}" if used is not None else ""
            deps = json.loads(b['depends_on']) if b.get('depends_on') else []
            deps_str = f"\nDepends on: {', '.join(deps)}" if deps else ""
            parts.append(
                f"\n### {b['name']} ({b['category']}) — used {b.get('usage_count', 0)} times{perf}"
            )
            parts.append(f"{b['description']}{deps_str}")
            parts.append(f"```python\n{b['code']}\n```")
        return "\n".join(parts)

    def _fmt_block_impact(self, performance: list[dict]) -> str:
        if not performance:
            return "No data yet."
        parts = []
        for b in performance[:10]:
            impact = b.get('impact', 0)
            sign = '+' if impact > 0 else ''
            parts.append(f"- **{b['name']}**: impact {sign}{impact:.3f} (used {b.get('usage_count', 0)}x)")
        return "\n".join(parts)

    # ── Parsers ─────────────────────────────────────────────────

    def _parse_strategies(self, content: str, model: str) -> list[GeneratedStrategy]:
        json_match = re.search(r'\[[\s\S]*\]', content)
        if not json_match:
            log.error(f"Failed to parse strategies: {content[:200]}")
            return []
        try:
            raw = json.loads(json_match.group())
        except json.JSONDecodeError as e:
            log.error(f"JSON parse error: {e}")
            return []

        strategies = []
        for item in raw:
            if not isinstance(item, dict) or "code" not in item:
                continue

            tf = item.get("primary_timeframe", "5m")
            if tf not in VALID_TIMEFRAMES:
                log.warning(f"Invalid timeframe '{tf}', defaulting to '5m'")
                tf = "5m"

            strategies.append(GeneratedStrategy(
                code=item["code"],
                description=item.get("description", ""),
                model_used=model,
                primary_timeframe=tf,
                eval_bars=item.get("eval_bars", 300),
                warmup_bars=item.get("warmup_bars", 0),
                candle_interval_seconds=item.get("candle_interval_seconds"),
                blocks_used=item.get("blocks_used"),
            ))
        return strategies

    def _parse_analysis(self, content: str) -> dict:
        json_match = re.search(r'\{[\s\S]*\}', content)
        if not json_match:
            return {"analysis": content, "insights": [], "next_directions": [], "plan": ""}
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            return {"analysis": content, "insights": [], "next_directions": [], "plan": ""}
