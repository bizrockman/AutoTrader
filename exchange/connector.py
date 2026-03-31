from __future__ import annotations

import asyncio
from typing import Protocol

import ccxt.async_support as ccxt


class Exchange(Protocol):
    """Interface that both live and paper trading implement."""

    async def get_price(self, symbol: str) -> float: ...
    async def get_ohlcv(self, symbol: str, timeframe: str = "1m", limit: int = 100) -> list[list]: ...
    async def place_order(self, symbol: str, side: str, quantity: float) -> dict: ...
    async def get_balance(self) -> dict: ...
    async def get_position(self, symbol: str) -> dict: ...
    async def get_orderbook(self, symbol: str, limit: int = 20) -> dict: ...
    async def get_funding_rate(self, symbol: str) -> dict: ...
    async def get_open_interest(self, symbol: str) -> dict: ...
    async def get_recent_trades(self, symbol: str, limit: int = 100) -> list[dict]: ...
    async def get_long_short_ratio(self, symbol: str, timeframe: str = "1h") -> list[dict]: ...


class BinanceConnector:
    """Fetches real market data from Binance."""

    def __init__(self, api_key: str = "", secret: str = "", timeout_sec: int = 15):
        self._exchange = ccxt.binance({
            "apiKey": api_key,
            "secret": secret,
            "enableRateLimit": True,
        })
        self._timeout = timeout_sec

    async def get_price(self, symbol: str) -> float:
        ticker = await asyncio.wait_for(self._exchange.fetch_ticker(symbol), timeout=self._timeout)
        return float(ticker["last"])

    async def get_ohlcv(self, symbol: str, timeframe: str = "1m", limit: int = 100) -> list[list]:
        return await asyncio.wait_for(
            self._exchange.fetch_ohlcv(symbol, timeframe, limit=limit), timeout=self._timeout
        )

    async def get_orderbook(self, symbol: str, limit: int = 20) -> dict:
        ob = await asyncio.wait_for(self._exchange.fetch_order_book(symbol, limit=limit), timeout=self._timeout)
        return {
            "bids": ob.get("bids", []),  # [[price, qty], ...]
            "asks": ob.get("asks", []),
            "spread": ob["asks"][0][0] - ob["bids"][0][0] if ob.get("asks") and ob.get("bids") else 0,
            "timestamp": ob.get("timestamp"),
        }

    async def get_funding_rate(self, symbol: str) -> dict:
        """Current funding rate — sentiment indicator for futures."""
        try:
            fr = await asyncio.wait_for(self._exchange.fetch_funding_rate(symbol), timeout=self._timeout)
            return {
                "funding_rate": fr.get("fundingRate"),
                "mark_price": fr.get("markPrice"),
                "index_price": fr.get("indexPrice"),
                "next_funding_time": fr.get("fundingDatetime"),
            }
        except Exception:
            return {"funding_rate": None, "error": "not_available_for_spot"}

    async def get_open_interest(self, symbol: str) -> dict:
        """Total open interest — how much is positioned."""
        try:
            oi = await asyncio.wait_for(self._exchange.fetch_open_interest(symbol), timeout=self._timeout)
            return {
                "open_interest": oi.get("openInterestAmount"),
                "open_interest_value": oi.get("openInterestValue"),
                "timestamp": oi.get("timestamp"),
            }
        except Exception:
            return {"open_interest": None, "error": "not_available_for_spot"}

    async def get_recent_trades(self, symbol: str, limit: int = 100) -> list[dict]:
        """Recent trades — volume microstructure."""
        trades = await asyncio.wait_for(self._exchange.fetch_trades(symbol, limit=limit), timeout=self._timeout)
        return [
            {
                "price": t["price"],
                "amount": t["amount"],
                "side": t["side"],
                "timestamp": t["timestamp"],
            }
            for t in trades
        ]

    async def get_long_short_ratio(self, symbol: str, timeframe: str = "1h", limit: int = 10) -> list[dict]:
        """Long/short account ratio — retail vs smart money."""
        try:
            data = await asyncio.wait_for(
                self._exchange.fetch_long_short_ratio_history(symbol, timeframe, limit=limit), timeout=self._timeout
            )
            return [
                {
                    "long_account": d.get("longAccount"),
                    "short_account": d.get("shortAccount"),
                    "long_short_ratio": d.get("longShortRatio"),
                    "timestamp": d.get("timestamp"),
                }
                for d in data
            ]
        except Exception:
            return []

    async def close(self) -> None:
        await self._exchange.close()
