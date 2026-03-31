"""Strategy Runner — executes generated strategies in Docker containers.

Each strategy runs in its own container with:
- Full Python environment (can pip install, use any library)
- No access to host filesystem or API keys
- Communication via stdin/stdout JSON lines
- Timeout and resource limits
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)


class _SafeEncoder(json.JSONEncoder):
    """Handles numpy / non-standard types on the host side."""
    def default(self, obj):
        try:
            import numpy as np
            if isinstance(obj, (np.bool_,)):
                return bool(obj)
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
        except ImportError:
            pass
        return super().default(obj)


def _host_dumps(obj: object) -> str:
    return json.dumps(obj, cls=_SafeEncoder)


# The wrapper script that runs inside the Docker container.
# It loads the strategy, receives ticks via stdin, sends signals via stdout.
CONTAINER_WORKER = r'''
import asyncio
import json
import os
import sys
import importlib.util

class _NumpySafeEncoder(json.JSONEncoder):
    """Handles numpy types that the standard json module rejects."""
    def default(self, obj):
        try:
            import numpy as np
            if isinstance(obj, (np.bool_,)):
                return bool(obj)
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
        except ImportError:
            pass
        return super().default(obj)

def _dumps(obj):
    return json.dumps(obj, cls=_NumpySafeEncoder)

class ExchangeProxy:
    """Proxy that communicates with the host via stdout/stdin."""

    def __init__(self):
        self._request_id = 0
        self._pending = {}

    async def _request(self, method, **kwargs):
        self._request_id += 1
        req = {"type": "exchange_request", "id": self._request_id, "method": method, "kwargs": kwargs}
        print(_dumps(req), flush=True)
        # Read response from stdin
        while True:
            line = await asyncio.get_event_loop().run_in_executor(None, sys.stdin.readline)
            if not line:
                raise ConnectionError("Host disconnected")
            msg = json.loads(line.strip())
            if msg.get("type") == "exchange_response" and msg.get("id") == self._request_id:
                if "error" in msg:
                    raise RuntimeError(msg["error"])
                return msg["result"]

    async def get_price(self, symbol):
        return await self._request("get_price", symbol=symbol)

    async def get_ohlcv(self, symbol, timeframe="1m", limit=100):
        return await self._request("get_ohlcv", symbol=symbol, timeframe=timeframe, limit=limit)

    async def place_order(self, symbol, side, quantity):
        return await self._request("place_order", symbol=symbol, side=side, quantity=quantity)

    async def get_balance(self):
        return await self._request("get_balance")

    async def get_position(self, symbol):
        return await self._request("get_position", symbol=symbol)

    async def get_orderbook(self, symbol, limit=20):
        return await self._request("get_orderbook", symbol=symbol, limit=limit)

    async def get_funding_rate(self, symbol):
        return await self._request("get_funding_rate", symbol=symbol)

    async def get_open_interest(self, symbol):
        return await self._request("get_open_interest", symbol=symbol)

    async def get_recent_trades(self, symbol, limit=100):
        return await self._request("get_recent_trades", symbol=symbol, limit=limit)

    async def get_long_short_ratio(self, symbol, timeframe="1h"):
        return await self._request("get_long_short_ratio", symbol=symbol, timeframe=timeframe)


async def main():
    # Load strategy from /strategy/strategy.py
    spec = importlib.util.spec_from_file_location("strategy_module", "/strategy/strategy.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    exchange = ExchangeProxy()
    # Pass timeframe from environment (set by host) to Strategy.__init__
    timeframe = os.environ.get("STRATEGY_TIMEFRAME", "5m")
    try:
        strategy = mod.Strategy(exchange, timeframe=timeframe)
    except TypeError:
        # Fallback for strategies that don't accept timeframe yet
        strategy = mod.Strategy(exchange)

    # Signal ready
    print(_dumps({"type": "ready"}), flush=True)

    # Main loop: receive ticks, call strategy
    while True:
        line = await asyncio.get_event_loop().run_in_executor(None, sys.stdin.readline)
        if not line:
            break
        msg = json.loads(line.strip())

        if msg["type"] == "tick":
            try:
                result = await strategy.on_tick(msg["symbol"], msg["price"], msg["timestamp"])
                state = strategy.get_state()
                print(_dumps({"type": "signal", "result": result, "state": state}), flush=True)
            except Exception as e:
                print(_dumps({"type": "error", "error": str(e)}), flush=True)

        elif msg["type"] == "candle":
            try:
                result = await strategy.on_candle(msg["symbol"], msg["candle"], msg["timestamp"])
                state = strategy.get_state()
                print(_dumps({"type": "signal", "result": result, "state": state}), flush=True)
            except Exception as e:
                print(_dumps({"type": "error", "error": str(e)}), flush=True)

        elif msg["type"] == "shutdown":
            break

    print(_dumps({"type": "shutdown_ack"}), flush=True)


asyncio.run(main())
'''


@dataclass
class StrategyProcess:
    """A running strategy in a Docker container."""
    strategy_id: str
    process: asyncio.subprocess.Process
    container_name: str
    temp_dir: str


class StrategyRunner:
    """Manages strategy execution in Docker containers."""

    def __init__(
        self,
        docker_image: str = "autotrader-strategy",
        docker_memory: str = "256m",
        docker_cpus: str = "0.5",
        ready_timeout_sec: int = 30,
        tick_timeout_sec: int = 10,
    ):
        self._docker_image = docker_image
        self._docker_memory = docker_memory
        self._docker_cpus = docker_cpus
        self._ready_timeout_sec = ready_timeout_sec
        self._tick_timeout_sec = tick_timeout_sec
        self._running: dict[str, StrategyProcess] = {}

    async def start_strategy(self, strategy_id: str, code: str, exchange, timeframe: str = "5m") -> StrategyProcess:
        """Start a strategy in a Docker container."""
        # Write strategy code and worker to a temp directory
        temp_dir = tempfile.mkdtemp(prefix=f"strat_{strategy_id}_")
        strategy_file = os.path.join(temp_dir, "strategy.py")
        worker_file = os.path.join(temp_dir, "worker.py")

        with open(strategy_file, "w") as f:
            f.write(code)
        with open(worker_file, "w") as f:
            f.write(CONTAINER_WORKER)

        container_name = f"strat_{strategy_id}"

        # Start Docker container
        # Mount temp_dir as /strategy, run worker.py
        proc = await asyncio.create_subprocess_exec(
            "docker", "run",
            "--rm",
            "--name", container_name,
            "-i",  # Keep stdin open
            "-e", f"STRATEGY_TIMEFRAME={timeframe}",
            "--network", "none",
            "--memory", self._docker_memory,
            "--cpus", self._docker_cpus,
            "-v", f"{temp_dir}:/strategy:ro",
            self._docker_image,
            "python", "/strategy/worker.py",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        sp = StrategyProcess(
            strategy_id=strategy_id,
            process=proc,
            container_name=container_name,
            temp_dir=temp_dir,
        )
        self._running[strategy_id] = sp

        # Wait for ready signal
        try:
            ready_line = await asyncio.wait_for(proc.stdout.readline(), timeout=self._ready_timeout_sec)
            msg = json.loads(ready_line.decode().strip())
            if msg.get("type") != "ready":
                raise RuntimeError(f"Strategy {strategy_id} did not send ready signal: {msg}")
        except asyncio.TimeoutError:
            await self.stop_strategy(strategy_id)
            raise RuntimeError(f"Strategy {strategy_id} timed out during startup")

        log.info(f"Strategy {strategy_id} started in container {container_name}")
        return sp

    async def send_tick(self, strategy_id: str, symbol: str, price: float, timestamp: float, exchange) -> dict | None:
        """Send a price tick to a running strategy and get its response."""
        sp = self._running.get(strategy_id)
        if not sp or sp.process.returncode is not None:
            return None

        tick_msg = _host_dumps({"type": "tick", "symbol": symbol, "price": price, "timestamp": timestamp}) + "\n"
        sp.process.stdin.write(tick_msg.encode())
        await sp.process.stdin.drain()

        # Read response (may include exchange requests)
        return await self._handle_strategy_output(sp, exchange)

    async def send_candle(self, strategy_id: str, symbol: str, candle: dict, timestamp: float, exchange) -> dict | None:
        """Send a candle to a running strategy."""
        sp = self._running.get(strategy_id)
        if not sp or sp.process.returncode is not None:
            return None

        msg = _host_dumps({"type": "candle", "symbol": symbol, "candle": candle, "timestamp": timestamp}) + "\n"
        sp.process.stdin.write(msg.encode())
        await sp.process.stdin.drain()

        return await self._handle_strategy_output(sp, exchange)

    async def _handle_strategy_output(self, sp: StrategyProcess, exchange) -> dict | None:
        """Read strategy output, handle exchange proxy requests."""
        try:
            while True:
                line = await asyncio.wait_for(sp.process.stdout.readline(), timeout=self._tick_timeout_sec)
                if not line:
                    return None

                try:
                    msg = json.loads(line.decode().strip())
                except json.JSONDecodeError:
                    log.warning(f"Strategy {sp.strategy_id} sent invalid JSON: {line.decode()[:100]}")
                    return {"type": "error", "error": "invalid_json"}

                if msg["type"] == "exchange_request":
                    # Forward exchange request to real exchange
                    method = msg["method"]
                    kwargs = msg["kwargs"]
                    try:
                        result = await getattr(exchange, method)(**kwargs)
                        response = {"type": "exchange_response", "id": msg["id"], "result": result}
                    except Exception as e:
                        response = {"type": "exchange_response", "id": msg["id"], "error": str(e)}

                    resp_line = _host_dumps(response) + "\n"
                    sp.process.stdin.write(resp_line.encode())
                    await sp.process.stdin.drain()
                    continue

                if msg["type"] == "signal":
                    return msg

                if msg["type"] == "error":
                    log.warning(f"Strategy {sp.strategy_id} error: {msg['error']}")
                    return msg

        except asyncio.TimeoutError:
            log.warning(f"Strategy {sp.strategy_id} timed out on tick")
            return {"type": "error", "error": "timeout"}

    async def stop_strategy(self, strategy_id: str) -> None:
        """Stop a running strategy."""
        sp = self._running.pop(strategy_id, None)
        if not sp:
            return

        try:
            if sp.process.returncode is None:
                shutdown_msg = _host_dumps({"type": "shutdown"}) + "\n"
                sp.process.stdin.write(shutdown_msg.encode())
                await sp.process.stdin.drain()
                await asyncio.wait_for(sp.process.wait(), timeout=5)
        except Exception:
            pass

        if sp.process.returncode is None:
            sp.process.kill()
            await sp.process.wait()

        # Clean up temp directory
        import shutil
        try:
            shutil.rmtree(sp.temp_dir, ignore_errors=True)
        except Exception as e:
            log.warning(f"Failed to clean up {sp.temp_dir}: {e}")

        log.info(f"Strategy {strategy_id} stopped")

    async def stop_all(self) -> None:
        """Stop all running strategies."""
        ids = list(self._running.keys())
        await asyncio.gather(*(self.stop_strategy(sid) for sid in ids))

    def is_running(self, strategy_id: str) -> bool:
        sp = self._running.get(strategy_id)
        return sp is not None and sp.process.returncode is None
