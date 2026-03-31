"""Integration tests — BinanceConnector against the real public API.

Every test here hits Binance over the network.
Run selectively:  pytest -m integration
"""
import pytest


pytestmark = pytest.mark.integration


async def test_get_price_returns_positive_float(real_connector):
    price = await real_connector.get_price("BTC/USDT")
    assert isinstance(price, float)
    assert price > 1000, f"BTC price {price} seems unrealistically low"
    assert price < 500_000, f"BTC price {price} seems unrealistically high"


async def test_get_ohlcv_returns_candles(real_connector):
    candles = await real_connector.get_ohlcv("BTC/USDT", "1m", limit=5)
    assert isinstance(candles, list)
    assert len(candles) >= 3, f"Expected at least 3 candles, got {len(candles)}"

    c = candles[-1]
    assert len(c) >= 6, "Candle should have [timestamp, open, high, low, close, volume]"
    ts, o, h, l, cl, vol = c[0], c[1], c[2], c[3], c[4], c[5]
    assert ts > 1_000_000_000_000, "Timestamp should be in milliseconds"
    assert h >= l, f"High {h} should be >= Low {l}"
    assert h >= o, f"High {h} should be >= Open {o}"
    assert vol >= 0, "Volume should be non-negative"


async def test_get_orderbook_has_bids_and_asks(real_connector):
    ob = await real_connector.get_orderbook("BTC/USDT", limit=5)
    assert "bids" in ob and "asks" in ob
    assert len(ob["bids"]) > 0, "Orderbook should have bids"
    assert len(ob["asks"]) > 0, "Orderbook should have asks"
    assert ob["spread"] >= 0, f"Spread should be non-negative, got {ob['spread']}"

    best_bid = ob["bids"][0][0]
    best_ask = ob["asks"][0][0]
    assert best_ask > best_bid, f"Ask {best_ask} should be > Bid {best_bid}"


async def test_get_recent_trades_returns_trades(real_connector):
    trades = await real_connector.get_recent_trades("BTC/USDT", limit=10)
    assert isinstance(trades, list)
    assert len(trades) >= 5, f"Expected trades, got {len(trades)}"

    t = trades[0]
    assert "price" in t and "amount" in t and "side" in t
    assert t["side"] in ("buy", "sell")
    assert t["price"] > 0
    assert t["amount"] > 0


async def test_get_funding_rate_returns_data_or_not_available(real_connector):
    fr = await real_connector.get_funding_rate("BTC/USDT")
    assert isinstance(fr, dict)
    if "error" not in fr:
        assert fr["funding_rate"] is not None
        assert isinstance(fr["funding_rate"], float)


async def test_get_open_interest_returns_data_or_not_available(real_connector):
    oi = await real_connector.get_open_interest("BTC/USDT")
    assert isinstance(oi, dict)
