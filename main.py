"""AutoTrader — Self-Evolving Crypto Trading Bot.

A dead-simple system that generates complexity through iteration:
1. LLM generates trading strategies as Python code
2. Strategies run live against real crypto markets (paper trading)
3. Results are evaluated, analyzed, and fed back
4. Next generation evolves from the best — 24/7, autonomously

Usage:
    uv run python main.py
"""
import asyncio
import logging
import sys

from config import Config
from evolution.orchestrator import Orchestrator


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("autotrader.log"),
        ],
    )


async def main() -> None:
    setup_logging()
    log = logging.getLogger("autotrader")

    log.info("=" * 60)
    log.info("  AutoTrader — Self-Evolving Crypto Trading Bot")
    log.info("  The system is simple. The complexity emerges.")
    log.info("=" * 60)

    config = Config()

    log.info(f"Symbol: {config.default_symbol}")
    log.info(f"Default model: {config.default_model}")
    log.info(f"Model pool: {config.model_pool}")
    log.info(f"Max parallel strategies: {config.max_parallel_strategies}")
    log.info(f"Initial balance: ${config.initial_balance:,.2f}")

    orchestrator = Orchestrator(config)
    await orchestrator.start()


if __name__ == "__main__":
    asyncio.run(main())
