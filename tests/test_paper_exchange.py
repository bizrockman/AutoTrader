"""Unit tests for the PaperExchange — no network calls needed.

Uses a mock connector that returns fixed prices.
Tests the core trading logic: buy, sell, reject, balance tracking.
"""
import pytest

from exchange.paper import PaperExchange


class MockConnector:
    """Returns a fixed price; no network needed."""

    def __init__(self, price: float = 70_000.0):
        self._price = price
        self._orderbook = {
            "bids": [[price - 5, 1.0], [price - 10, 2.0]],
            "asks": [[price + 5, 1.0], [price + 10, 2.0]],
            "spread": 10.0,
            "timestamp": 1711900000000,
        }

    async def get_price(self, symbol: str) -> float:
        return self._price

    async def get_ohlcv(self, symbol, timeframe="1m", limit=100):
        return []

    async def get_orderbook(self, symbol, limit=20):
        return self._orderbook

    async def get_funding_rate(self, symbol):
        return {"funding_rate": 0.0001}

    async def get_open_interest(self, symbol):
        return {"open_interest": None}

    async def get_recent_trades(self, symbol, limit=100):
        return []

    async def get_long_short_ratio(self, symbol, timeframe="1h"):
        return []

    def set_price(self, price: float):
        self._price = price


pytestmark = pytest.mark.asyncio


@pytest.fixture
def exchange():
    return PaperExchange(
        connector=MockConnector(70_000.0),
        initial_balance=10_000.0,
        fee_pct=0.1,
    )


async def test_initial_balance(exchange):
    balance = await exchange.get_balance()
    assert balance["quote_balance"] == 10_000.0
    assert balance["total_value"] == 10_000.0
    assert balance["pnl"] == 0.0


async def test_buy_reduces_balance(exchange):
    result = await exchange.place_order("BTC/USDT", "buy", 0.01)
    assert result["status"] == "filled"
    assert result["price"] == 70_000.0
    assert result["quantity"] == 0.01

    balance = await exchange.get_balance()
    expected_cost = 70_000.0 * 0.01  # 700
    expected_fee = expected_cost * 0.001  # 0.70
    assert balance["quote_balance"] == pytest.approx(10_000 - expected_cost - expected_fee, rel=1e-6)


async def test_sell_increases_balance(exchange):
    await exchange.place_order("BTC/USDT", "buy", 0.01)
    result = await exchange.place_order("BTC/USDT", "sell", 0.01)
    assert result["status"] == "filled"

    balance = await exchange.get_balance()
    # After buy+sell at same price, we lose 2x fee
    total_fees = 70_000.0 * 0.01 * 0.001 * 2  # 1.40
    assert balance["total_value"] == pytest.approx(10_000 - total_fees, rel=1e-4)


async def test_buy_rejected_insufficient_balance(exchange):
    result = await exchange.place_order("BTC/USDT", "buy", 1.0)
    # 1 BTC = 70,000 USDT > 10,000 balance
    assert result["status"] == "rejected"
    assert "insufficient" in result["reason"]


async def test_sell_rejected_no_position(exchange):
    result = await exchange.place_order("BTC/USDT", "sell", 0.01)
    assert result["status"] == "rejected"
    assert "insufficient" in result["reason"]


async def test_sell_rejected_partial_position(exchange):
    await exchange.place_order("BTC/USDT", "buy", 0.01)
    result = await exchange.place_order("BTC/USDT", "sell", 0.02)
    assert result["status"] == "rejected"


async def test_unknown_side_rejected(exchange):
    result = await exchange.place_order("BTC/USDT", "short", 0.01)
    assert result["status"] == "rejected"
    assert "unknown" in result["reason"]


async def test_position_tracking(exchange):
    await exchange.place_order("BTC/USDT", "buy", 0.05)

    pos = await exchange.get_position("BTC/USDT")
    assert pos["quantity"] == 0.05
    assert pos["avg_entry_price"] == 70_000.0


async def test_no_position_returns_zero(exchange):
    pos = await exchange.get_position("ETH/USDT")
    assert pos["quantity"] == 0
    assert pos["avg_entry_price"] == 0


async def test_trade_history_records_trades(exchange):
    await exchange.place_order("BTC/USDT", "buy", 0.01)
    await exchange.place_order("BTC/USDT", "sell", 0.01)

    history = exchange.get_trade_history()
    assert len(history) == 2
    assert history[0]["side"] == "buy"
    assert history[1]["side"] == "sell"


async def test_reset_clears_everything(exchange):
    await exchange.place_order("BTC/USDT", "buy", 0.01)
    exchange.reset()

    balance = await exchange.get_balance()
    assert balance["quote_balance"] == 10_000.0
    assert len(exchange.get_trade_history()) == 0


async def test_fee_is_applied(exchange):
    result = await exchange.place_order("BTC/USDT", "buy", 0.01)
    assert result["fee"] > 0
    expected_fee = 70_000.0 * 0.01 * 0.001
    assert result["fee"] == pytest.approx(expected_fee, rel=1e-6)


async def test_multiple_buys_average_entry(exchange):
    exchange._connector.set_price(70_000.0)
    await exchange.place_order("BTC/USDT", "buy", 0.01)

    exchange._connector.set_price(72_000.0)
    await exchange.place_order("BTC/USDT", "buy", 0.01)

    pos = await exchange.get_position("BTC/USDT")
    assert pos["quantity"] == 0.02
    assert pos["avg_entry_price"] == pytest.approx(71_000.0, rel=1e-6)
