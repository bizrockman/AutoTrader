"""Loop Detector — prevents the evolution from going in circles.

Uses structural (AST) comparison for duplicate detection and
description-based concept tracking for exploration diversity.
No hardcoded technique lists — the system stays open to novel approaches.
"""
from __future__ import annotations

import ast
import logging
import re
from collections import Counter
from dataclasses import dataclass

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
        return re.sub(r'\s+', ' ', code).strip()

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

    tokens1 = set(norm1.split())
    tokens2 = set(norm2.split())

    if not tokens1 or not tokens2:
        return 0.0

    intersection = tokens1 & tokens2
    union = tokens1 | tokens2

    return len(intersection) / len(union)


def _extract_concepts(description: str) -> set[str]:
    """Extract key concepts from a strategy description.

    Open-ended: uses the LLM's own words, not a fixed taxonomy.
    Extracts meaningful multi-word and single-word concepts.
    """
    if not description:
        return set()

    text = description.lower()
    # Remove filler words but keep domain-specific terms
    stop = {
        "a", "an", "the", "this", "that", "is", "are", "was", "were", "be",
        "been", "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "can", "shall", "must",
        "and", "or", "but", "if", "then", "else", "when", "while", "for",
        "to", "from", "in", "on", "at", "by", "with", "of", "as", "into",
        "it", "its", "we", "our", "they", "their", "i", "my", "you", "your",
        "not", "no", "so", "very", "more", "most", "also", "just", "than",
        "each", "every", "all", "both", "such", "uses", "using", "based",
        "strategy", "trading", "trades", "trade", "market", "price", "prices",
    }
    words = re.findall(r'[a-z][a-z_-]+', text)
    return {w for w in words if w not in stop and len(w) > 2}


def _ast_fingerprint(code: str) -> str:
    """Structural fingerprint of code: what node types appear and in what pattern."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return ""

    node_types = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.If, ast.Compare, ast.BoolOp, ast.Call,
                             ast.Attribute, ast.BinOp, ast.For, ast.While)):
            node_types.append(type(node).__name__)

    return " ".join(sorted(node_types))


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
        """Detect A→B→A→B oscillation patterns using structural fingerprints."""
        if len(recent_strategies) < 4:
            return PingPongCheck(detected=False, pattern_description="Not enough data.", oscillating_concepts=[])

        recent = recent_strategies[:window]

        # Use AST fingerprints — structural, not name-based
        fingerprints = []
        for s in recent:
            fp = _ast_fingerprint(s.get("code", ""))
            fingerprints.append(fp if fp else "unparseable")

        transitions = []
        for i in range(len(fingerprints) - 1):
            transitions.append((fingerprints[i], fingerprints[i + 1]))

        transition_counts = Counter(transitions)
        repeated = {t: c for t, c in transition_counts.items() if c >= 2}

        if repeated:
            # Pull concept names from descriptions for the warning message
            oscillating = set()
            for s in recent:
                oscillating.update(_extract_concepts(s.get("description", "")))

            return PingPongCheck(
                detected=True,
                pattern_description=(
                    f"Detected structural oscillation over {len(recent)} strategies. "
                    f"Repeated transitions: {len(repeated)}. "
                    f"The system keeps producing structurally similar code patterns."
                ),
                oscillating_concepts=sorted(list(oscillating)[:10]),
            )

        return PingPongCheck(detected=False, pattern_description="No oscillation detected.", oscillating_concepts=[])

    def get_exploration_summary(self, all_strategies: list[dict]) -> str:
        """Summarize what has been explored — open-ended, no fixed technique list.

        Uses the LLM's own descriptions to map the exploration space.
        Encourages novelty without suggesting from a predefined menu.
        """
        if not all_strategies:
            return (
                "No strategies tested yet. You have complete freedom — "
                "explore any approach you think could work. "
                "Traditional indicators, novel math, microstructure, "
                "or something entirely new."
            )

        # Build concept frequency from descriptions (LLM's own vocabulary)
        concept_counts: Counter[str] = Counter()
        for s in all_strategies:
            concepts = _extract_concepts(s.get("description", ""))
            concept_counts.update(concepts)

        explored = sorted(concept_counts.items(), key=lambda x: -x[1])

        # Structural diversity: how many distinct AST patterns exist?
        fingerprints = set()
        for s in all_strategies:
            fp = _ast_fingerprint(s.get("code", ""))
            if fp:
                fingerprints.add(fp)
        structural_diversity = len(fingerprints) / max(len(all_strategies), 1)

        summary = "## Exploration Map\n\n"
        summary += f"Total strategies tested: {len(all_strategies)}\n"
        summary += f"Structural diversity: {structural_diversity:.0%} "
        summary += f"({len(fingerprints)} distinct code patterns)\n\n"

        if explored:
            summary += "### Concepts explored (from your descriptions):\n"
            for name, count in explored[:15]:
                summary += f"- {name}: {count}x\n"

        if structural_diversity < 0.5:
            summary += (
                "\n**Low structural diversity** — many strategies share similar code patterns. "
                "Consider fundamentally different algorithmic structures, not just different parameters.\n"
            )

        summary += (
            "\nYou are NOT limited to known indicators. Invent new approaches: "
            "combine data sources in novel ways, use statistical methods, "
            "exploit microstructure — anything that might work.\n"
        )

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
            parts.append(
                f"WARNING: Your last proposed strategy was {loop_check.similarity_score:.0%} similar to "
                f"'{loop_check.most_similar_id}'. That approach scored fitness={loop_check.previous_fitness}. "
                f"Do NOT repeat it. Try something fundamentally different."
            )

        if ping_pong and ping_pong.detected:
            parts.append(
                f"WARNING: Oscillation detected! {ping_pong.pattern_description} "
                f"Break the cycle — try a completely unrelated approach."
            )

        parts.append(exploration_summary)

        return "\n\n".join(parts) if parts else ""
