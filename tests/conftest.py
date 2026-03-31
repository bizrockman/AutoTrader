"""Shared fixtures for all test modules.

Fixtures marked with `integration` create real network connections
to Binance public endpoints — no API key required.
"""
import pytest
import pytest_asyncio

from exchange.connector import BinanceConnector
from exchange.paper import PaperExchange


@pytest_asyncio.fixture
async def real_connector():
    """Live BinanceConnector against the real public API."""
    c = BinanceConnector()
    yield c
    await c.close()


@pytest_asyncio.fixture
async def real_exchange(real_connector):
    """PaperExchange backed by a live BinanceConnector."""
    return PaperExchange(
        connector=real_connector,
        initial_balance=10_000.0,
        fee_pct=0.1,
    )
