from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

from exchange.connector import BinanceConnector

log = logging.getLogger(__name__)


@dataclass
class Position:
    symbol: str
    quantity: float = 0.0
    avg_entry_price: float = 0.0
    unrealized_pnl: float = 0.0


@dataclass
class Trade:
    symbol: str
    side: str
    quantity: float
    price: float
    fee: float
    timestamp: str


class PaperExchange:
    """Paper trading engine using real market prices from Binance.

    Implements the same interface as a live exchange connector.
    Strategies see no difference between paper and live.
    """

    def __init__(
        self,
        connector: BinanceConnector,
        initial_balance: float = 10_000.0,
        fee_pct: float = 0.1,
        quote_currency: str = "USDT",
    ):
        self._connector = connector
        self._fee_pct = fee_pct / 100  # 0.1% → 0.001
        self._quote_currency = quote_currency

        self._quote_balance: float = initial_balance
        self._positions: dict[str, Position] = {}
        self._trades: list[Trade] = []
        self._initial_balance = initial_balance

    # ── Exchange Interface ──────────────────────────────────────

    async def get_price(self, symbol: str) -> float:
        return await self._connector.get_price(symbol)

    async def get_ohlcv(self, symbol: str, timeframe: str = "1m", limit: int = 100) -> list[list]:
        return await self._connector.get_ohlcv(symbol, timeframe, limit)

    async def place_order(self, symbol: str, side: str, quantity: float) -> dict:
        price = await self.get_price(symbol)
        cost = price * quantity
        fee = cost * self._fee_pct

        if side == "buy":
            total_cost = cost + fee
            if total_cost > self._quote_balance:
                return {"status": "rejected", "reason": "insufficient balance", "balance": self._quote_balance}

            self._quote_balance -= total_cost
            pos = self._positions.setdefault(symbol, Position(symbol=symbol))
            # Update average entry price
            total_qty = pos.quantity + quantity
            if total_qty > 0:
                pos.avg_entry_price = (pos.avg_entry_price * pos.quantity + price * quantity) / total_qty
            pos.quantity = total_qty

        elif side == "sell":
            pos = self._positions.get(symbol)
            if not pos or pos.quantity < quantity:
                available = pos.quantity if pos else 0.0
                return {"status": "rejected", "reason": "insufficient position", "available": available}

            revenue = cost - fee
            self._quote_balance += revenue
            pos.quantity -= quantity
            if pos.quantity == 0:
                pos.avg_entry_price = 0.0
        else:
            return {"status": "rejected", "reason": f"unknown side: {side}"}

        trade = Trade(
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            fee=fee,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        self._trades.append(trade)
        log.info(f"Paper {side.upper()} {quantity} {symbol} @ {price:.2f} (fee: {fee:.4f})")

        return {
            "status": "filled",
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "price": price,
            "fee": fee,
            "timestamp": trade.timestamp,
        }

    async def get_balance(self) -> dict:
        # Calculate total portfolio value
        total_value = self._quote_balance
        positions_value = 0.0
        for pos in self._positions.values():
            if pos.quantity > 0:
                try:
                    price = await self.get_price(pos.symbol)
                    positions_value += pos.quantity * price
                except Exception:
                    pass
        total_value += positions_value

        return {
            "quote_balance": self._quote_balance,
            "positions_value": positions_value,
            "total_value": total_value,
            "initial_balance": self._initial_balance,
            "pnl": total_value - self._initial_balance,
            "pnl_pct": ((total_value / self._initial_balance) - 1) * 100 if self._initial_balance > 0 else 0,
            self._quote_currency: {"free": self._quote_balance, "used": positions_value, "total": total_value},
        }

    async def get_position(self, symbol: str) -> dict:
        pos = self._positions.get(symbol)
        if not pos or pos.quantity == 0:
            return {"symbol": symbol, "quantity": 0, "avg_entry_price": 0, "unrealized_pnl": 0}

        try:
            current_price = await self.get_price(symbol)
            unrealized = (current_price - pos.avg_entry_price) * pos.quantity
        except Exception:
            current_price = 0.0
            unrealized = 0.0

        return {
            "symbol": symbol,
            "quantity": pos.quantity,
            "avg_entry_price": pos.avg_entry_price,
            "current_price": current_price,
            "unrealized_pnl": unrealized,
        }

    # ── Market data passthrough (read-only, no paper tracking) ──

    async def get_orderbook(self, symbol: str, limit: int = 20) -> dict:
        return await self._connector.get_orderbook(symbol, limit)

    async def get_funding_rate(self, symbol: str) -> dict:
        return await self._connector.get_funding_rate(symbol)

    async def get_open_interest(self, symbol: str) -> dict:
        return await self._connector.get_open_interest(symbol)

    async def get_recent_trades(self, symbol: str, limit: int = 100) -> list[dict]:
        return await self._connector.get_recent_trades(symbol, limit)

    async def get_long_short_ratio(self, symbol: str, timeframe: str = "1h") -> list[dict]:
        return await self._connector.get_long_short_ratio(symbol, timeframe)

    # ── Paper-specific helpers ──────────────────────────────────

    def get_trade_history(self) -> list[dict]:
        return [
            {"symbol": t.symbol, "side": t.side, "quantity": t.quantity,
             "price": t.price, "fee": t.fee, "timestamp": t.timestamp}
            for t in self._trades
        ]

    def reset(self) -> None:
        self._quote_balance = self._initial_balance
        self._positions.clear()
        self._trades.clear()
