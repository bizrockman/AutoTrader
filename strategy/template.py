"""Strategy interface contract.

Every generated strategy must implement this interface.
The LLM receives this as part of its generation prompt.
"""

STRATEGY_TEMPLATE = '''
class Strategy:
    """Trading strategy interface.

    Args:
        exchange: Proxy object with methods:
            # Core
            - await get_price(symbol) -> float
            - await get_ohlcv(symbol, timeframe, limit) -> list[list]
            - await place_order(symbol, side, quantity) -> dict
            - await get_balance() -> dict
            - await get_position(symbol) -> dict

            # Market depth
            - await get_orderbook(symbol, limit=20) -> dict
              Returns: {"bids": [[price, qty], ...], "asks": [...], "spread": float}

            # Derivatives data (returns empty/None for spot)
            - await get_funding_rate(symbol) -> dict
              Returns: {"funding_rate": float, "mark_price": float, "next_funding_time": str}
            - await get_open_interest(symbol) -> dict
              Returns: {"open_interest": float, "open_interest_value": float}
            - await get_long_short_ratio(symbol, timeframe="1h") -> list[dict]
              Returns: [{"long_account": float, "short_account": float, "long_short_ratio": float}]

            # Trade flow
            - await get_recent_trades(symbol, limit=100) -> list[dict]
              Returns: [{"price": float, "amount": float, "side": str, "timestamp": int}]

    The exchange object operates on real market data with paper trading.
    Orders are executed at the current market price with a 0.1% fee.
    """

    def __init__(self, exchange):
        self.exchange = exchange
        # Initialize your indicators, state, etc. here

    async def on_tick(self, symbol: str, price: float, timestamp: float) -> dict | None:
        """Called every ~5 seconds with the current price.

        Return None for no action, or a dict:
            {"action": "buy" | "sell", "symbol": str, "quantity": float, "reason": str}
        """
        return None

    async def on_candle(self, symbol: str, candle: dict, timestamp: float) -> dict | None:
        """Called when a new candle closes.

        candle keys: open, high, low, close, volume, timestamp

        Return None for no action, or a dict:
            {"action": "buy" | "sell", "symbol": str, "quantity": float, "reason": str}
        """
        return None

    def get_state(self) -> dict:
        """Return internal state for logging and analysis."""
        return {}
'''


STRATEGY_INTERFACE_DOC = """
## Strategy Interface

Your strategy must be a single Python file containing a `Strategy` class with this exact interface:

```python
{template}
```

### Rules:
1. The class MUST be named `Strategy`
2. `__init__` receives an `exchange` object — store it as `self.exchange`
3. `on_tick` and `on_candle` are async — use `await` for exchange calls
4. Return `None` for no action, or a trade dict with action, symbol, quantity, reason
5. `get_state` returns a dict with any internal state you want logged
6. You may import: math, statistics, collections, json, dataclasses, typing, asyncio
7. You may use any Python standard library module
8. You may use numpy, pandas, ta if needed (available in the container)
9. Keep your strategy in a SINGLE file — all logic in one class

### Available Market Data:
- **get_price / get_ohlcv**: Basic price and candle data (any timeframe)
- **get_orderbook**: Bid/ask depth — see where liquidity sits, detect walls
- **get_funding_rate**: Futures funding rate — sentiment indicator
- **get_open_interest**: Total positioned — crowding indicator
- **get_recent_trades**: Trade flow — volume microstructure, buy/sell pressure
- **get_long_short_ratio**: Account ratio — retail vs smart money positioning

Use what you need. Not every strategy needs all data sources.
""".format(template=STRATEGY_TEMPLATE)
