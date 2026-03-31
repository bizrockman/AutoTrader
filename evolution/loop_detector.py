"""Loop Detector — prevents the evolution from going in circles.

Neither Karpathy's autoresearch nor Meta-Harness have this.
This is our differentiator.
"""
from __future__ import annotations

import ast
import logging
import re
from collections import Counter
from dataclasses import dataclass

import litellm

log = logging.getLogger(__name__)


@dataclass
class LoopCheck:
    is_duplicate: bool
    similarity_score: float
    most_similar_id: str | None
    most_similar_description: str | None
    previous_fitness: float | None
    recommendation: str


@dataclass
class PingPongCheck:
    detected: bool
    pattern_description: str
    oscillating_concepts: list[str]


def _normalize_code(code: str) -> str:
    """Normalize code for comparison: remove comments, whitespace, docstrings."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        # If code doesn't parse, fall back to simple normalization
        return re.sub(r'\s+', ' ', code).strip()

    # Remove docstrings
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
            if (node.body and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, (ast.Constant, ast.Str))):
                node.body.pop(0)

    return ast.dump(tree)


def _code_similarity(code1: str, code2: str) -> float:
    """Compute structural similarity between two strategy codes using AST comparison."""
    norm1 = _normalize_code(code1)
    norm2 = _normalize_code(code2)

    if norm1 == norm2:
        return 1.0

    # Token-based Jaccard similarity on the AST dump
    tokens1 = set(norm1.split())
    tokens2 = set(norm2.split())

    if not tokens1 or not tokens2:
        return 0.0

    intersection = tokens1 & tokens2
    union = tokens1 | tokens2

    return len(intersection) / len(union)


def _extract_indicators(code: str) -> set[str]:
    """Extract indicator/technique names from strategy code."""
    indicators = set()
    patterns = [
        r'\brsi\b', r'\bmacd\b', r'\bema\b', r'\bsma\b', r'\bbollinger\b',
        r'\bvwap\b', r'\batr\b', r'\bstochastic\b', r'\bmomentum\b',
        r'\bmean.?reversion\b', r'\bbreakout\b', r'\btrend.?follow\b',
        r'\bscalp\b', r'\bgrid\b', r'\barbitrage\b', r'\bvolume\b',
        r'\bvolatility\b', r'\bsupport\b', r'\bresistance\b',
        r'\bfibonacci\b', r'\bichimoku\b', r'\bdivergence\b',
    ]
    code_lower = code.lower()
    for pattern in patterns:
        if re.search(pattern, code_lower):
            indicators.add(re.sub(r'[\\\.?]', '', pattern.strip(r'\b')))
    return indicators


class LoopDetector:
    """Detects when the evolution agent is going in circles."""

    def __init__(self, similarity_threshold: float = 0.7):
        self.similarity_threshold = similarity_threshold

    def check_similarity(
        self,
        new_code: str,
        new_description: str,
        existing_strategies: list[dict],
    ) -> LoopCheck:
        """Check if a new strategy is too similar to existing ones."""
        if not existing_strategies:
            return LoopCheck(
                is_duplicate=False, similarity_score=0.0,
                most_similar_id=None, most_similar_description=None,
                previous_fitness=None, recommendation="First strategy — no comparison needed.",
            )

        best_sim = 0.0
        best_match = None

        for existing in existing_strategies:
            sim = _code_similarity(new_code, existing.get("code", ""))
            if sim > best_sim:
                best_sim = sim
                best_match = existing

        is_dup = best_sim >= self.similarity_threshold

        if is_dup and best_match:
            recommendation = (
                f"Strategy is {best_sim:.0%} similar to '{best_match.get('id', '?')}' "
                f"({best_match.get('description', 'N/A')}). "
                f"Previous fitness: {best_match.get('fitness_score', 'N/A')}. "
                f"Try a fundamentally different approach."
            )
        else:
            recommendation = "Strategy appears sufficiently novel."

        return LoopCheck(
            is_duplicate=is_dup,
            similarity_score=best_sim,
            most_similar_id=best_match.get("id") if best_match else None,
            most_similar_description=best_match.get("description") if best_match else None,
            previous_fitness=best_match.get("fitness_score") if best_match else None,
            recommendation=recommendation,
        )

    def detect_ping_pong(self, recent_strategies: list[dict], window: int = 10) -> PingPongCheck:
        """Detect A→B→A→B oscillation patterns in recent strategies."""
        if len(recent_strategies) < 4:
            return PingPongCheck(detected=False, pattern_description="Not enough data.", oscillating_concepts=[])

        recent = recent_strategies[:window]

        # Extract indicator sets for each strategy
        indicator_sequence = []
        for s in recent:
            indicators = _extract_indicators(s.get("code", ""))
            indicator_sequence.append(frozenset(indicators) if indicators else frozenset({"unknown"}))

        # Check for oscillation: do we see the same concepts alternating?
        if len(indicator_sequence) < 4:
            return PingPongCheck(detected=False, pattern_description="Insufficient sequence.", oscillating_concepts=[])

        # Simple oscillation check: look at consecutive pairs
        transitions = []
        for i in range(len(indicator_sequence) - 1):
            transitions.append((indicator_sequence[i], indicator_sequence[i + 1]))

        # Count transitions — if same transition appears multiple times, might be ping-pong
        transition_counts = Counter(transitions)
        repeated = {t: c for t, c in transition_counts.items() if c >= 2}

        if repeated:
            oscillating = set()
            for (a, b), count in repeated.items():
                oscillating.update(a)
                oscillating.update(b)

            return PingPongCheck(
                detected=True,
                pattern_description=f"Detected oscillation between concept groups over {len(recent)} strategies. "
                                    f"Repeated transitions: {len(repeated)}",
                oscillating_concepts=sorted(oscillating),
            )

        return PingPongCheck(detected=False, pattern_description="No oscillation detected.", oscillating_concepts=[])

    def get_exploration_summary(self, all_strategies: list[dict]) -> str:
        """Summarize what has been explored and suggest new directions."""
        if not all_strategies:
            return "No strategies tested yet. Start with diverse approaches: momentum, mean-reversion, breakout, volatility-based."

        # Collect all indicators used
        all_indicators: Counter[str] = Counter()
        for s in all_strategies:
            indicators = _extract_indicators(s.get("code", ""))
            all_indicators.update(indicators)

        explored = sorted(all_indicators.items(), key=lambda x: -x[1])

        known_techniques = {
            "rsi", "macd", "ema", "sma", "bollinger", "vwap", "atr",
            "stochastic", "momentum", "mean_reversion", "breakout",
            "trend_follow", "scalp", "grid", "arbitrage", "volume",
            "volatility", "support", "resistance", "fibonacci", "ichimoku", "divergence",
        }
        used = {name for name, _ in explored}
        unused = known_techniques - used

        summary = "## Exploration Map\n\n"
        summary += "### Well-explored:\n"
        for name, count in explored[:10]:
            summary += f"- {name}: tested {count} times\n"

        if unused:
            summary += "\n### Unexplored techniques:\n"
            for name in sorted(unused):
                summary += f"- {name}\n"

        summary += f"\n### Total strategies tested: {len(all_strategies)}"
        return summary

    def build_loop_warning(
        self,
        loop_check: LoopCheck | None,
        ping_pong: PingPongCheck | None,
        exploration_summary: str,
    ) -> str:
        """Build a warning message for the generator if loops are detected."""
        parts = []

        if loop_check and loop_check.is_duplicate:
            parts.append(f"WARNING: Your last proposed strategy was {loop_check.similarity_score:.0%} similar to "
                        f"'{loop_check.most_similar_id}'. That approach scored fitness={loop_check.previous_fitness}. "
                        f"Do NOT repeat it. Try something fundamentally different.")

        if ping_pong and ping_pong.detected:
            parts.append(f"WARNING: Ping-pong pattern detected! You are oscillating between: "
                        f"{', '.join(ping_pong.oscillating_concepts)}. "
                        f"Break the cycle — try a completely unrelated approach.")

        parts.append(exploration_summary)

        return "\n\n".join(parts) if parts else ""
