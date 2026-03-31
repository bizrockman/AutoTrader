"""Tests for the PaperExchange.

Unit tests use a MockConnector (no network).
Integration tests (marked @pytest.mark.integration) use a real BinanceConnector.
"""
import pytest

from exchange.paper import PaperExchange


# ── Mock for unit tests ─────────────────────────────────────────────


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


@pytest.fixture
def exchange():
    return PaperExchange(
        connector=MockConnector(70_000.0),
        initial_balance=10_000.0,
        fee_pct=0.1,
    )


# ── Unit tests (mock connector, fast) ───────────────────────────────


async def test_initial_balance(exchange):
    balance = await exchange.get_balance()
    assert balance["quote_balance"] == 10_000.0
    assert balance["total_value"] == 10_000.0
    assert balance["pnl"] == 0.0
    # ccxt-compatible keys
    assert balance["USDT"]["free"] == 10_000.0
    assert balance["USDT"]["total"] == 10_000.0


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
    total_fees = 70_000.0 * 0.01 * 0.001 * 2  # 1.40
    assert balance["total_value"] == pytest.approx(10_000 - total_fees, rel=1e-4)


async def test_buy_rejected_insufficient_balance(exchange):
    result = await exchange.place_order("BTC/USDT", "buy", 1.0)
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


# ── Integration tests (real Binance API) ────────────────────────────


@pytest.mark.integration
async def test_integration_buy_sell_roundtrip(real_exchange):
    """Full buy→sell cycle against live market prices."""
    # Balance before
    bal_before = await real_exchange.get_balance()
    assert bal_before["USDT"]["free"] == 10_000.0

    # Buy a small amount of BTC
    buy = await real_exchange.place_order("BTC/USDT", "buy", 0.001)
    assert buy["status"] == "filled"
    assert buy["price"] > 1000
    assert buy["fee"] > 0

    # Position should exist
    pos = await real_exchange.get_position("BTC/USDT")
    assert pos["quantity"] == 0.001
    assert pos["avg_entry_price"] == buy["price"]

    # Balance should have decreased
    bal_mid = await real_exchange.get_balance()
    assert bal_mid["USDT"]["free"] < 10_000.0
    assert bal_mid["positions_value"] > 0

    # Sell it back
    sell = await real_exchange.place_order("BTC/USDT", "sell", 0.001)
    assert sell["status"] == "filled"

    # Position should be zero
    pos_after = await real_exchange.get_position("BTC/USDT")
    assert pos_after["quantity"] == 0

    # Balance after: slightly less than 10k due to fees + potential price move
    bal_after = await real_exchange.get_balance()
    assert bal_after["USDT"]["free"] > 9_000  # shouldn't lose more than 10% on 0.001 BTC
    assert bal_after["USDT"]["free"] < 10_100  # shouldn't gain more than 1% on 0.001 BTC

    # Trade history has exactly 2 entries
    history = real_exchange.get_trade_history()
    assert len(history) == 2
    assert history[0]["side"] == "buy"
    assert history[1]["side"] == "sell"


@pytest.mark.integration
async def test_integration_balance_ccxt_compat(real_exchange):
    """get_balance() returns both internal and ccxt-compatible keys."""
    balance = await real_exchange.get_balance()

    # Internal keys
    assert "quote_balance" in balance
    assert "total_value" in balance
    assert "pnl" in balance
    assert "pnl_pct" in balance

    # ccxt-compatible keys (what LLM-generated strategies use)
    assert "USDT" in balance
    assert "free" in balance["USDT"]
    assert "used" in balance["USDT"]
    assert "total" in balance["USDT"]
    assert balance["USDT"]["free"] == balance["quote_balance"]


@pytest.mark.integration
async def test_integration_orderbook_passthrough(real_exchange):
    """PaperExchange passes orderbook requests to the real connector."""
    ob = await real_exchange.get_orderbook("BTC/USDT", limit=5)
    assert len(ob["bids"]) > 0
    assert len(ob["asks"]) > 0
    assert ob["asks"][0][0] > ob["bids"][0][0]


@pytest.mark.integration
async def test_integration_ohlcv_passthrough(real_exchange):
    """PaperExchange passes OHLCV requests to the real connector."""
    candles = await real_exchange.get_ohlcv("BTC/USDT", "1m", limit=3)
    assert len(candles) >= 2
    assert candles[-1][2] >= candles[-1][3]  # high >= low


@pytest.mark.integration
async def test_integration_reject_too_large_order(real_exchange):
    """Cannot buy more than the balance allows at real prices."""
    result = await real_exchange.place_order("BTC/USDT", "buy", 10.0)
    assert result["status"] == "rejected"
    assert "insufficient" in result["reason"]
