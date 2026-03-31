from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone

import aiosqlite

log = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS strategies (
    id              TEXT PRIMARY KEY,
    code            TEXT NOT NULL,
    description     TEXT,
    parent_ids      TEXT DEFAULT '[]',
    model_used      TEXT,
    wave_id         INTEGER,
    mutation_type   TEXT CHECK(mutation_type IN ('novel', 'mutation', 'crossover', 'survivor')),
    embedding       TEXT,
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS strategy_runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_id     TEXT NOT NULL REFERENCES strategies(id),
    started_at      TEXT NOT NULL DEFAULT (datetime('now')),
    ended_at        TEXT,
    eval_period_sec INTEGER NOT NULL,
    status          TEXT NOT NULL DEFAULT 'running'
                    CHECK(status IN ('running', 'completed', 'crashed', 'killed')),
    error_message   TEXT,
    start_snapshot  TEXT,
    end_snapshot    TEXT
);

CREATE TABLE IF NOT EXISTS trades (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_id     TEXT NOT NULL REFERENCES strategies(id),
    run_id          INTEGER NOT NULL REFERENCES strategy_runs(id),
    symbol          TEXT NOT NULL,
    side            TEXT NOT NULL CHECK(side IN ('buy', 'sell')),
    quantity        REAL NOT NULL,
    price           REAL NOT NULL,
    fee             REAL NOT NULL DEFAULT 0,
    reason          TEXT,
    timestamp       TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS metrics (
    run_id          INTEGER NOT NULL REFERENCES strategy_runs(id),
    strategy_id     TEXT NOT NULL REFERENCES strategies(id),
    pnl             REAL,
    pnl_pct         REAL,
    sharpe          REAL,
    max_drawdown    REAL,
    trade_count     INTEGER DEFAULT 0,
    win_rate        REAL,
    crash_count     INTEGER DEFAULT 0,
    fitness_score   REAL,
    PRIMARY KEY (run_id, strategy_id)
);

CREATE TABLE IF NOT EXISTS insights (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    wave_id         INTEGER,
    category        TEXT NOT NULL CHECK(category IN ('pattern', 'failure', 'market_regime', 'technique', 'meta')),
    content         TEXT NOT NULL,
    confidence      REAL DEFAULT 0.5,
    model_used      TEXT,
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS evolution_waves (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at      TEXT NOT NULL DEFAULT (datetime('now')),
    ended_at        TEXT,
    trigger_reason  TEXT,
    strategies_evaluated TEXT DEFAULT '[]',
    analysis        TEXT,
    decisions       TEXT,
    plan            TEXT,
    model_used      TEXT
);

CREATE TABLE IF NOT EXISTS loop_checks (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_id     TEXT NOT NULL REFERENCES strategies(id),
    similarity_score REAL,
    most_similar_id TEXT,
    action_taken    TEXT,
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Hall of Fame: strategies that proved themselves and can be recalled later
CREATE TABLE IF NOT EXISTS hall_of_fame (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_id     TEXT NOT NULL REFERENCES strategies(id),
    promoted_at     TEXT NOT NULL DEFAULT (datetime('now')),
    market_regime   TEXT,
    tags            TEXT DEFAULT '[]',
    summary         TEXT,
    best_pnl_pct    REAL,
    best_sharpe     REAL,
    total_runs      INTEGER DEFAULT 1,
    avg_fitness     REAL,
    status          TEXT NOT NULL DEFAULT 'active'
                    CHECK(status IN ('active', 'retired', 'champion'))
);

-- Building blocks: the DNA of the system.
-- IMMUTABLE: rows are never updated. A new version = a new row.
-- This guarantees cold-starts and reruns from any wave work correctly.
CREATE TABLE IF NOT EXISTS blocks (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT NOT NULL,
    version         INTEGER NOT NULL DEFAULT 1,
    code            TEXT NOT NULL,
    description     TEXT NOT NULL,
    category        TEXT NOT NULL
                    CHECK(category IN ('indicator', 'signal', 'filter', 'risk', 'utility', 'composition', 'strategy')),
    depends_on      TEXT DEFAULT '[]',
    origin_strategy TEXT REFERENCES strategies(id),
    origin_wave     INTEGER,
    -- Performance stats (updated in-place — these are aggregates, not history)
    usage_count     INTEGER DEFAULT 0,
    avg_fitness_when_used   REAL,
    avg_fitness_when_unused REAL,
    best_regime     TEXT,
    worst_regime    TEXT,
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(name, version)
);

-- Track which block VERSION each strategy used — the "genome"
CREATE TABLE IF NOT EXISTS strategy_blocks (
    strategy_id     TEXT NOT NULL REFERENCES strategies(id),
    block_name      TEXT NOT NULL,
    block_version   INTEGER NOT NULL,
    parameters      TEXT DEFAULT '{}',
    PRIMARY KEY (strategy_id, block_name)
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class EvolutionContext:
    """Everything the LLM needs to generate the next wave of strategies."""
    top_strategies: list[dict]
    recent_insights: list[dict]
    failed_approaches: list[dict]
    hall_of_fame: list[dict]
    blocks: list[dict]
    block_performance: list[dict]
    current_wave: int
    total_strategies_tested: int
    market_summary: dict | None = None


class KnowledgeStore:
    def __init__(self, db_path: str = "autotrader.db"):
        self.db_path = db_path
        self._db: aiosqlite.Connection | None = None

    async def connect(self) -> None:
        self._db = await aiosqlite.connect(self.db_path)
        self._db.row_factory = aiosqlite.Row
        await self._db.executescript(SCHEMA)
        await self._db.commit()

    async def close(self) -> None:
        if self._db:
            await self._db.close()

    @property
    def db(self) -> aiosqlite.Connection:
        assert self._db is not None, "Call connect() first"
        return self._db

    # ── Strategies ──────────────────────────────────────────────

    async def save_strategy(
        self,
        strategy_id: str,
        code: str,
        description: str,
        model_used: str,
        wave_id: int | None = None,
    ) -> None:
        await self.db.execute(
            """INSERT INTO strategies (id, code, description, model_used, wave_id, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (strategy_id, code, description, model_used, wave_id, _now()),
        )
        await self.db.commit()

    async def get_strategy(self, strategy_id: str) -> dict | None:
        async with self.db.execute("SELECT * FROM strategies WHERE id = ?", (strategy_id,)) as cur:
            row = await cur.fetchone()
            return dict(row) if row else None

    # ── Strategy Runs ───────────────────────────────────────────

    async def start_run(self, strategy_id: str, eval_period_sec: int, start_snapshot: dict | None = None) -> int:
        cur = await self.db.execute(
            """INSERT INTO strategy_runs (strategy_id, eval_period_sec, started_at, status, start_snapshot)
               VALUES (?, ?, ?, 'running', ?)""",
            (strategy_id, eval_period_sec, _now(), json.dumps(start_snapshot) if start_snapshot else None),
        )
        await self.db.commit()
        return cur.lastrowid  # type: ignore[return-value]

    async def finish_run(
        self, run_id: int, status: str = "completed",
        error_message: str | None = None, end_snapshot: dict | None = None,
    ) -> None:
        await self.db.execute(
            """UPDATE strategy_runs SET ended_at = ?, status = ?, error_message = ?, end_snapshot = ?
               WHERE id = ?""",
            (_now(), status, error_message, json.dumps(end_snapshot) if end_snapshot else None, run_id),
        )
        await self.db.commit()

    async def get_running_strategies(self) -> list[dict]:
        async with self.db.execute(
            "SELECT sr.*, s.code, s.description FROM strategy_runs sr JOIN strategies s ON sr.strategy_id = s.id WHERE sr.status = 'running'"
        ) as cur:
            return [dict(r) for r in await cur.fetchall()]

    # ── Trades ──────────────────────────────────────────────────

    async def record_trade(
        self,
        strategy_id: str,
        run_id: int,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        fee: float = 0.0,
        reason: str | None = None,
    ) -> None:
        await self.db.execute(
            """INSERT INTO trades (strategy_id, run_id, symbol, side, quantity, price, fee, reason, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (strategy_id, run_id, symbol, side, quantity, price, fee, reason, _now()),
        )
        await self.db.commit()

    async def get_trades_for_run(self, run_id: int) -> list[dict]:
        async with self.db.execute("SELECT * FROM trades WHERE run_id = ? ORDER BY timestamp", (run_id,)) as cur:
            return [dict(r) for r in await cur.fetchall()]

    # ── Metrics ─────────────────────────────────────────────────

    async def save_metrics(self, run_id: int, strategy_id: str, metrics: dict) -> None:
        await self.db.execute(
            """INSERT OR REPLACE INTO metrics (run_id, strategy_id, pnl, pnl_pct, sharpe, max_drawdown,
               trade_count, win_rate, crash_count, fitness_score)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (run_id, strategy_id, metrics.get("pnl"), metrics.get("pnl_pct"),
             metrics.get("sharpe"), metrics.get("max_drawdown"),
             metrics.get("trade_count", 0), metrics.get("win_rate"),
             metrics.get("crash_count", 0), metrics.get("fitness_score")),
        )
        await self.db.commit()

    # ── Insights ────────────────────────────────────────────────

    async def save_insight(
        self, wave_id: int | None, category: str, content: str,
        confidence: float = 0.5, model_used: str | None = None,
    ) -> None:
        await self.db.execute(
            "INSERT INTO insights (wave_id, category, content, confidence, model_used, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (wave_id, category, content, confidence, model_used, _now()),
        )
        await self.db.commit()

    async def get_recent_insights(self, limit: int = 20) -> list[dict]:
        async with self.db.execute(
            "SELECT * FROM insights ORDER BY created_at DESC LIMIT ?", (limit,)
        ) as cur:
            return [dict(r) for r in await cur.fetchall()]

    # ── Evolution Waves ─────────────────────────────────────────

    async def start_wave(self, trigger_reason: str = "scheduled") -> int:
        cur = await self.db.execute(
            "INSERT INTO evolution_waves (started_at, trigger_reason) VALUES (?, ?)",
            (_now(), trigger_reason),
        )
        await self.db.commit()
        return cur.lastrowid  # type: ignore[return-value]

    async def save_wave_plan(self, wave_id: int, plan: str) -> None:
        await self.db.execute(
            "UPDATE evolution_waves SET plan = ? WHERE id = ?", (plan, wave_id),
        )
        await self.db.commit()

    async def get_latest_plan(self) -> str:
        """Get the plan from the most recent completed wave."""
        async with self.db.execute(
            "SELECT plan FROM evolution_waves WHERE plan IS NOT NULL ORDER BY id DESC LIMIT 1"
        ) as cur:
            row = await cur.fetchone()
            return row[0] if row else ""

    async def finish_wave(
        self, wave_id: int, strategies_evaluated: list[str],
        analysis: str, decisions: dict, model_used: str,
    ) -> None:
        await self.db.execute(
            """UPDATE evolution_waves
               SET ended_at = ?, strategies_evaluated = ?, analysis = ?, decisions = ?, model_used = ?
               WHERE id = ?""",
            (_now(), json.dumps(strategies_evaluated), analysis, json.dumps(decisions), model_used, wave_id),
        )
        await self.db.commit()

    async def get_current_wave_id(self) -> int:
        async with self.db.execute("SELECT MAX(id) FROM evolution_waves") as cur:
            row = await cur.fetchone()
            return row[0] or 0

    # ── Building Blocks (IMMUTABLE — never update, always append) ──

    async def save_block(
        self,
        name: str,
        code: str,
        description: str,
        category: str,
        depends_on: list[str] | None = None,
        origin_strategy: str | None = None,
        origin_wave: int | None = None,
    ) -> int:
        """Save a building block. If name exists, creates a new version. Never overwrites.

        Returns the version number.
        """
        # Get the next version number for this block name
        async with self.db.execute(
            "SELECT MAX(version) FROM blocks WHERE name = ?", (name,)
        ) as cur:
            row = await cur.fetchone()
            version = (row[0] or 0) + 1

        await self.db.execute(
            """INSERT INTO blocks (name, version, code, description, category,
               depends_on, origin_strategy, origin_wave, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (name, version, code, description, category,
             json.dumps(depends_on or []), origin_strategy, origin_wave, _now()),
        )
        await self.db.commit()
        log.info(f"Block '{name}' v{version} saved (origin: wave {origin_wave})")
        return version

    async def get_blocks(self, category: str | None = None) -> list[dict]:
        """Get the LATEST version of each block. Most-used first."""
        query = """
            SELECT b.* FROM blocks b
            INNER JOIN (SELECT name, MAX(version) as max_v FROM blocks GROUP BY name) latest
            ON b.name = latest.name AND b.version = latest.max_v
        """
        if category:
            query += " WHERE b.category = ?"
            async with self.db.execute(query + " ORDER BY b.usage_count DESC", (category,)) as cur:
                return [dict(r) for r in await cur.fetchall()]
        async with self.db.execute(query + " ORDER BY b.usage_count DESC") as cur:
            return [dict(r) for r in await cur.fetchall()]

    async def get_blocks_at_wave(self, wave_id: int) -> list[dict]:
        """Get blocks as they existed at a specific wave — for reruns."""
        async with self.db.execute(
            """SELECT b.* FROM blocks b
               INNER JOIN (
                   SELECT name, MAX(version) as max_v FROM blocks
                   WHERE origin_wave <= ? GROUP BY name
               ) snapshot
               ON b.name = snapshot.name AND b.version = snapshot.max_v
               ORDER BY b.usage_count DESC""",
            (wave_id,),
        ) as cur:
            return [dict(r) for r in await cur.fetchall()]

    async def get_block_history(self, name: str) -> list[dict]:
        """Get all versions of a block — its evolution over time."""
        async with self.db.execute(
            "SELECT * FROM blocks WHERE name = ? ORDER BY version ASC", (name,)
        ) as cur:
            return [dict(r) for r in await cur.fetchall()]

    async def record_block_usage(self, strategy_id: str, block_name: str, parameters: dict | None = None) -> None:
        """Record that a strategy uses the latest version of a block."""
        # Find the latest version
        async with self.db.execute(
            "SELECT MAX(version) FROM blocks WHERE name = ?", (block_name,)
        ) as cur:
            row = await cur.fetchone()
            version = row[0] if row and row[0] else 1

        await self.db.execute(
            """INSERT OR IGNORE INTO strategy_blocks (strategy_id, block_name, block_version, parameters)
               VALUES (?, ?, ?, ?)""",
            (strategy_id, block_name, version, json.dumps(parameters or {})),
        )
        # Update usage count on the specific version
        await self.db.execute(
            "UPDATE blocks SET usage_count = usage_count + 1 WHERE name = ? AND version = ?",
            (block_name, version),
        )
        await self.db.commit()

    async def update_block_fitness(self, block_name: str, block_version: int, fitness: float, was_used: bool) -> None:
        """Update performance stats on a specific block version."""
        if was_used:
            await self.db.execute(
                """UPDATE blocks SET
                   avg_fitness_when_used = CASE
                       WHEN avg_fitness_when_used IS NULL THEN ?
                       ELSE (avg_fitness_when_used * usage_count + ?) / (usage_count + 1)
                   END
                   WHERE name = ? AND version = ?""",
                (fitness, fitness, block_name, block_version),
            )
        else:
            await self.db.execute(
                """UPDATE blocks SET
                   avg_fitness_when_unused = CASE
                       WHEN avg_fitness_when_unused IS NULL THEN ?
                       ELSE (avg_fitness_when_unused + ?) / 2
                   END
                   WHERE name = ? AND version = ?""",
                (fitness, fitness, block_name, block_version),
            )
        await self.db.commit()

    async def get_strategy_genome(self, strategy_id: str) -> list[dict]:
        """Get the exact block versions a strategy used — its 'genome'."""
        async with self.db.execute(
            """SELECT sb.block_name, sb.block_version, sb.parameters,
                      b.code, b.description, b.category
               FROM strategy_blocks sb
               JOIN blocks b ON sb.block_name = b.name AND sb.block_version = b.version
               WHERE sb.strategy_id = ?""",
            (strategy_id,),
        ) as cur:
            return [dict(r) for r in await cur.fetchall()]

    async def get_block_performance_report(self) -> list[dict]:
        """Which blocks (latest version) contribute to success?"""
        async with self.db.execute(
            """SELECT b.name, b.version, b.category, b.usage_count, b.description,
                      b.avg_fitness_when_used, b.avg_fitness_when_unused,
                      COALESCE(b.avg_fitness_when_used, 0) - COALESCE(b.avg_fitness_when_unused, 0) as impact
               FROM blocks b
               INNER JOIN (SELECT name, MAX(version) as max_v FROM blocks GROUP BY name) latest
               ON b.name = latest.name AND b.version = latest.max_v
               WHERE b.usage_count > 0
               ORDER BY impact DESC"""
        ) as cur:
            return [dict(r) for r in await cur.fetchall()]

    # ── Loop Checks ─────────────────────────────────────────────

    async def save_loop_check(
        self, strategy_id: str, similarity_score: float,
        most_similar_id: str | None, action_taken: str,
    ) -> None:
        await self.db.execute(
            "INSERT INTO loop_checks (strategy_id, similarity_score, most_similar_id, action_taken, created_at) VALUES (?, ?, ?, ?, ?)",
            (strategy_id, similarity_score, most_similar_id, action_taken, _now()),
        )
        await self.db.commit()

    # ── Hall of Fame ──────────────────────────────────────────────

    async def promote_to_hall_of_fame(
        self,
        strategy_id: str,
        market_regime: str | None = None,
        tags: list[str] | None = None,
        summary: str | None = None,
        best_pnl_pct: float | None = None,
        best_sharpe: float | None = None,
        avg_fitness: float | None = None,
    ) -> None:
        """Promote a proven strategy to the Hall of Fame."""
        await self.db.execute(
            """INSERT INTO hall_of_fame
               (strategy_id, market_regime, tags, summary, best_pnl_pct, best_sharpe, avg_fitness, promoted_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (strategy_id, market_regime, json.dumps(tags or []), summary,
             best_pnl_pct, best_sharpe, avg_fitness, _now()),
        )
        await self.db.commit()

    async def get_hall_of_fame(self, status: str = "active") -> list[dict]:
        """Get all Hall of Fame strategies with their code."""
        async with self.db.execute(
            """SELECT hof.*, s.code, s.description
               FROM hall_of_fame hof JOIN strategies s ON hof.strategy_id = s.id
               WHERE hof.status = ?
               ORDER BY hof.avg_fitness DESC""",
            (status,),
        ) as cur:
            return [dict(r) for r in await cur.fetchall()]

    async def get_hall_of_fame_for_regime(self, market_regime: str) -> list[dict]:
        """Get Hall of Fame strategies that worked in a specific market regime."""
        async with self.db.execute(
            """SELECT hof.*, s.code, s.description
               FROM hall_of_fame hof JOIN strategies s ON hof.strategy_id = s.id
               WHERE hof.status = 'active' AND hof.market_regime = ?
               ORDER BY hof.avg_fitness DESC""",
            (market_regime,),
        ) as cur:
            return [dict(r) for r in await cur.fetchall()]

    async def update_hall_of_fame_stats(self, strategy_id: str, new_fitness: float, new_pnl_pct: float) -> None:
        """Update stats when a Hall of Fame strategy gets re-tested."""
        await self.db.execute(
            """UPDATE hall_of_fame SET
               total_runs = total_runs + 1,
               avg_fitness = (avg_fitness * total_runs + ?) / (total_runs + 1),
               best_pnl_pct = MAX(best_pnl_pct, ?),
               best_sharpe = MAX(best_sharpe, ?)
               WHERE strategy_id = ?""",
            (new_fitness, new_pnl_pct, new_pnl_pct, strategy_id),
        )
        await self.db.commit()

    async def retire_from_hall_of_fame(self, strategy_id: str) -> None:
        """Retire a strategy that no longer performs."""
        await self.db.execute(
            "UPDATE hall_of_fame SET status = 'retired' WHERE strategy_id = ?",
            (strategy_id,),
        )
        await self.db.commit()

    # ── Evolution Context ───────────────────────────────────────

    async def get_evolution_context(self, top_k: int = 5, recent_insights_k: int = 10) -> EvolutionContext:
        # Top strategies by fitness (all-time, not just recent)
        async with self.db.execute(
            """SELECT s.id, s.code, s.description, m.fitness_score, m.pnl_pct, m.sharpe, m.win_rate
               FROM metrics m JOIN strategies s ON m.strategy_id = s.id
               ORDER BY m.fitness_score DESC LIMIT ?""",
            (top_k,),
        ) as cur:
            top_strategies = [dict(r) for r in await cur.fetchall()]

        # Recent insights
        insights = await self.get_recent_insights(recent_insights_k)

        # Failed approaches (worst fitness)
        async with self.db.execute(
            """SELECT s.id, s.description, m.fitness_score, m.pnl_pct
               FROM metrics m JOIN strategies s ON m.strategy_id = s.id
               WHERE m.fitness_score IS NOT NULL
               ORDER BY m.fitness_score ASC LIMIT ?""",
            (top_k,),
        ) as cur:
            failed = [dict(r) for r in await cur.fetchall()]

        # Hall of Fame — proven strategies the LLM can build on
        hof = await self.get_hall_of_fame()

        # Building blocks and their performance
        blocks = await self.get_blocks()
        block_performance = await self.get_block_performance_report()

        # Wave count
        wave_id = await self.get_current_wave_id()

        # Total strategies tested
        async with self.db.execute("SELECT COUNT(*) FROM strategies") as cur:
            row = await cur.fetchone()
            total = row[0] if row else 0

        return EvolutionContext(
            top_strategies=top_strategies,
            recent_insights=insights,
            failed_approaches=failed,
            hall_of_fame=hof,
            blocks=blocks,
            block_performance=block_performance,
            current_wave=wave_id,
            total_strategies_tested=total,
        )

    # ── All strategy codes (for loop detection) ─────────────────

    async def get_all_strategy_summaries(self) -> list[dict]:
        async with self.db.execute(
            "SELECT id, description, code FROM strategies ORDER BY created_at DESC"
        ) as cur:
            return [dict(r) for r in await cur.fetchall()]
