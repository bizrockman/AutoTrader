"""Meta-Tracker — monitors the health of the evolution process itself.

Answers: Is the system getting better? Is it stagnating? Which models work?
How much does it cost? Should we change approach?
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from knowledge.store import KnowledgeStore

log = logging.getLogger(__name__)


@dataclass
class EvolutionHealth:
    """Snapshot of how the evolution process is doing."""
    fitness_trend: float  # positive = improving, negative = degrading
    best_fitness_ever: float
    avg_fitness_recent: float
    waves_since_improvement: int
    total_waves: int
    total_strategies: int
    block_count: int
    hall_of_fame_count: int
    model_performance: dict[str, float]  # model → avg fitness
    is_stagnating: bool
    summary: str  # human-readable summary for the LLM

    def to_prompt_section(self) -> str:
        """Format for inclusion in LLM prompts."""
        parts = [f"## Evolution Health\n{self.summary}"]

        if self.model_performance:
            parts.append("\n### Model Performance:")
            for model, fitness in sorted(self.model_performance.items(), key=lambda x: -x[1]):
                parts.append(f"- **{model}**: avg fitness {fitness:.3f}")

        if self.is_stagnating:
            parts.append(
                f"\n### WARNING: Stagnation detected!"
                f"\nNo improvement in {self.waves_since_improvement} waves. "
                f"Try something radically different."
            )

        return "\n".join(parts)


class MetaTracker:
    """Tracks the health of the evolution process."""

    async def get_health(self, store: KnowledgeStore) -> EvolutionHealth:
        """Compute current evolution health metrics."""

        # Total counts
        total_waves = await store.get_current_wave_id()
        async with store.db.execute("SELECT COUNT(*) FROM strategies") as cur:
            total_strategies = (await cur.fetchone())[0]
        async with store.db.execute("SELECT COUNT(DISTINCT name) FROM blocks") as cur:
            block_count = (await cur.fetchone())[0]
        async with store.db.execute("SELECT COUNT(*) FROM hall_of_fame WHERE status = 'active'") as cur:
            hof_count = (await cur.fetchone())[0]

        # Best fitness ever
        async with store.db.execute("SELECT MAX(fitness_score) FROM metrics") as cur:
            row = await cur.fetchone()
            best_ever = row[0] if row and row[0] else 0.0

        # Fitness trend: compare last 5 waves vs previous 5
        recent_fitness = await self._avg_fitness_for_waves(store, total_waves - 4, total_waves)
        previous_fitness = await self._avg_fitness_for_waves(store, total_waves - 9, total_waves - 5)
        trend = recent_fitness - previous_fitness if previous_fitness else 0.0

        # Waves since improvement
        waves_since = await self._waves_since_improvement(store, best_ever)

        # Model performance
        model_perf = await self._model_performance(store)

        # Stagnation
        is_stagnating = waves_since >= 10

        # Build summary
        summary = self._build_summary(
            total_waves, total_strategies, block_count, hof_count,
            best_ever, recent_fitness, trend, waves_since, is_stagnating,
        )

        return EvolutionHealth(
            fitness_trend=round(trend, 4),
            best_fitness_ever=best_ever,
            avg_fitness_recent=recent_fitness,
            waves_since_improvement=waves_since,
            total_waves=total_waves,
            total_strategies=total_strategies,
            block_count=block_count,
            hall_of_fame_count=hof_count,
            model_performance=model_perf,
            is_stagnating=is_stagnating,
            summary=summary,
        )

    async def _avg_fitness_for_waves(self, store: KnowledgeStore, from_wave: int, to_wave: int) -> float:
        """Average fitness of strategies from a range of waves."""
        async with store.db.execute(
            """SELECT AVG(m.fitness_score)
               FROM metrics m JOIN strategies s ON m.strategy_id = s.id
               WHERE s.wave_id BETWEEN ? AND ? AND m.fitness_score IS NOT NULL""",
            (max(from_wave, 1), to_wave),
        ) as cur:
            row = await cur.fetchone()
            return row[0] if row and row[0] else 0.0

    async def _waves_since_improvement(self, store: KnowledgeStore, best_ever: float) -> int:
        """How many waves since we last set a new best fitness."""
        if best_ever <= 0:
            return 0
        async with store.db.execute(
            """SELECT s.wave_id FROM metrics m JOIN strategies s ON m.strategy_id = s.id
               WHERE m.fitness_score >= ? * 0.95
               ORDER BY s.wave_id DESC LIMIT 1""",
            (best_ever,),
        ) as cur:
            row = await cur.fetchone()
            if not row:
                return 0
            current_wave = await store.get_current_wave_id()
            return current_wave - row[0]

    async def _model_performance(self, store: KnowledgeStore) -> dict[str, float]:
        """Average fitness per model."""
        async with store.db.execute(
            """SELECT s.model_used, AVG(m.fitness_score), COUNT(*)
               FROM metrics m JOIN strategies s ON m.strategy_id = s.id
               WHERE m.fitness_score IS NOT NULL AND s.model_used IS NOT NULL
               GROUP BY s.model_used
               HAVING COUNT(*) >= 3"""
        ) as cur:
            rows = await cur.fetchall()
            return {row[0]: round(row[1], 4) for row in rows}

    @staticmethod
    def _build_summary(
        waves: int, strategies: int, blocks: int, hof: int,
        best: float, recent: float, trend: float, stagnation: int,
        is_stagnating: bool,
    ) -> str:
        parts = [
            f"Waves: {waves}, Strategies tested: {strategies}, "
            f"Blocks discovered: {blocks}, Hall of Fame: {hof}",
            f"Best fitness ever: {best:.3f}, Recent avg: {recent:.3f}, "
            f"Trend: {'+' if trend > 0 else ''}{trend:.3f}",
        ]
        if is_stagnating:
            parts.append(f"STAGNATING: No improvement in {stagnation} waves. Consider radical changes.")
        elif trend > 0.05:
            parts.append("Evolution is progressing well. Keep iterating.")
        elif trend < -0.05:
            parts.append("Performance is degrading. Review recent changes.")
        else:
            parts.append("Performance is flat. Try diversifying approaches.")
        return "\n".join(parts)
