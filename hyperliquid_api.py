"""
Hyperliquid API Client for fetching positions, market data, and liquidation information
"""

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

import aiohttp
import requests
from cachetools import TTLCache

from config import (
    HYPERLIQUID_API_URL,
    HYPERLIQUID_TESTNET_API_URL,
    TRACKED_ASSETS,
    ASSET_MAX_LEVERAGE,
    DEFAULT_LIQUIDATION_CONFIG,
)


@dataclass
class Position:
    """Represents a trading position"""

    address: str
    coin: str
    size: float  # Positive for long, negative for short
    entry_price: float
    mark_price: float
    liquidation_price: Optional[float]
    leverage: float
    margin_used: float
    position_value: float
    unrealized_pnl: float
    margin_type: str  # "cross" or "isolated"
    timestamp: datetime

    @property
    def is_long(self) -> bool:
        return self.size > 0

    @property
    def side(self) -> str:
        return "long" if self.is_long else "short"

    @property
    def abs_size(self) -> float:
        return abs(self.size)


@dataclass
class AssetInfo:
    """Asset metadata"""

    name: str
    sz_decimals: int
    max_leverage: int
    only_isolated: bool = False


@dataclass
class MarketData:
    """Current market data for an asset"""

    coin: str
    mark_price: float
    mid_price: float
    open_interest: float
    funding_rate: float
    volume_24h: float
    timestamp: datetime


class HyperliquidAPI:
    """
    Client for interacting with the Hyperliquid API

    Provides methods to fetch:
    - User positions and clearinghouse state
    - Market metadata and current prices
    - Liquidatable positions
    - Asset context information
    """

    def __init__(
        self,
        use_testnet: bool = False,
        cache_ttl: int = 60,
        rate_limit: float = 5.0,
    ):
        self.api_url = HYPERLIQUID_TESTNET_API_URL if use_testnet else HYPERLIQUID_API_URL
        self.cache = TTLCache(maxsize=1000, ttl=cache_ttl)
        self.rate_limit = rate_limit
        self.last_request_time = 0.0
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    def _rate_limit_wait(self):
        """Enforce rate limiting"""
        elapsed = time.time() - self.last_request_time
        min_interval = 1.0 / self.rate_limit
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self.last_request_time = time.time()

    def _post_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Make a synchronous POST request to the API"""
        self._rate_limit_wait()
        headers = {"Content-Type": "application/json"}
        response = requests.post(
            self.api_url,
            headers=headers,
            json=payload,
            timeout=DEFAULT_LIQUIDATION_CONFIG.request_timeout,
        )
        response.raise_for_status()
        return response.json()

    async def _async_post_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Make an asynchronous POST request to the API"""
        if not self.session:
            self.session = aiohttp.ClientSession()

        headers = {"Content-Type": "application/json"}
        async with self.session.post(
            self.api_url,
            headers=headers,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=DEFAULT_LIQUIDATION_CONFIG.request_timeout),
        ) as response:
            response.raise_for_status()
            return await response.json()

    def get_meta_and_asset_contexts(self) -> Tuple[List[AssetInfo], Dict[str, MarketData]]:
        """
        Fetch metadata and current context for all assets

        Returns:
            Tuple of (list of asset info, dict of market data by coin)
        """
        cache_key = "meta_and_asset_contexts"
        if cache_key in self.cache:
            return self.cache[cache_key]

        payload = {"type": "metaAndAssetCtxs"}
        response = self._post_request(payload)

        # Parse meta information
        meta = response[0]
        asset_ctxs = response[1]

        assets = []
        for universe_item in meta.get("universe", []):
            asset = AssetInfo(
                name=universe_item.get("name", ""),
                sz_decimals=universe_item.get("szDecimals", 0),
                max_leverage=universe_item.get("maxLeverage", 1),
                only_isolated=universe_item.get("onlyIsolated", False),
            )
            assets.append(asset)

        # Parse current market data
        market_data = {}
        for i, ctx in enumerate(asset_ctxs):
            if i < len(assets):
                coin = assets[i].name
                market_data[coin] = MarketData(
                    coin=coin,
                    mark_price=float(ctx.get("markPx", 0)),
                    mid_price=float(ctx.get("midPx", 0)),
                    open_interest=float(ctx.get("openInterest", 0)),
                    funding_rate=float(ctx.get("funding", 0)),
                    volume_24h=float(ctx.get("dayNtlVlm", 0)),
                    timestamp=datetime.now(),
                )

        result = (assets, market_data)
        self.cache[cache_key] = result
        return result

    def get_all_mids(self) -> Dict[str, float]:
        """
        Get mid prices for all assets

        Returns:
            Dictionary mapping coin name to mid price
        """
        cache_key = "all_mids"
        if cache_key in self.cache:
            return self.cache[cache_key]

        payload = {"type": "allMids"}
        response = self._post_request(payload)

        mids = {coin: float(price) for coin, price in response.items()}
        self.cache[cache_key] = mids
        return mids

    def get_user_state(self, address: str) -> Dict[str, Any]:
        """
        Get clearinghouse state for a specific user address

        Args:
            address: User's wallet address (42-character hex)

        Returns:
            Dictionary containing user's positions, margin info, etc.
        """
        payload = {"type": "clearinghouseState", "user": address}
        return self._post_request(payload)

    def get_user_positions(self, address: str) -> List[Position]:
        """
        Get parsed positions for a specific user

        Args:
            address: User's wallet address

        Returns:
            List of Position objects
        """
        state = self.get_user_state(address)
        positions = []

        # Get current prices for calculating mark price
        mids = self.get_all_mids()

        for asset_position in state.get("assetPositions", []):
            pos_data = asset_position.get("position", {})
            coin = pos_data.get("coin", "")

            if not coin:
                continue

            # Parse leverage info
            leverage_info = pos_data.get("leverage", {})
            leverage_type = leverage_info.get("type", "cross")
            leverage_value = float(leverage_info.get("value", 1))

            # Parse position data
            size = float(pos_data.get("szi", 0))
            entry_price = float(pos_data.get("entryPx", 0)) if pos_data.get("entryPx") else 0
            liq_price_str = pos_data.get("liquidationPx")
            liq_price = float(liq_price_str) if liq_price_str else None

            position = Position(
                address=address,
                coin=coin,
                size=size,
                entry_price=entry_price,
                mark_price=mids.get(coin, entry_price),
                liquidation_price=liq_price,
                leverage=leverage_value,
                margin_used=float(pos_data.get("marginUsed", 0)),
                position_value=float(pos_data.get("positionValue", 0)),
                unrealized_pnl=float(pos_data.get("unrealizedPnl", 0)),
                margin_type=leverage_type,
                timestamp=datetime.now(),
            )
            positions.append(position)

        return positions

    async def get_multiple_user_positions(
        self, addresses: List[str]
    ) -> Dict[str, List[Position]]:
        """
        Fetch positions for multiple users concurrently

        Args:
            addresses: List of wallet addresses

        Returns:
            Dictionary mapping address to list of positions
        """
        async def fetch_one(address: str) -> Tuple[str, List[Position]]:
            try:
                # Use sync version wrapped in executor for now
                positions = await asyncio.get_event_loop().run_in_executor(
                    None, self.get_user_positions, address
                )
                return address, positions
            except Exception as e:
                print(f"Error fetching positions for {address}: {e}")
                return address, []

        tasks = [fetch_one(addr) for addr in addresses]
        results = await asyncio.gather(*tasks)

        return {addr: positions for addr, positions in results}

    def get_open_orders(self, address: str) -> List[Dict[str, Any]]:
        """
        Get open orders for a user

        Args:
            address: User's wallet address

        Returns:
            List of open orders
        """
        payload = {"type": "openOrders", "user": address}
        return self._post_request(payload)

    def get_user_fills(
        self, address: str, start_time: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get trade fills for a user

        Args:
            address: User's wallet address
            start_time: Optional start timestamp in milliseconds

        Returns:
            List of trade fills
        """
        payload = {"type": "userFills", "user": address}
        if start_time:
            payload["startTime"] = start_time
        return self._post_request(payload)

    def get_funding_history(self, coin: str, start_time: int) -> List[Dict[str, Any]]:
        """
        Get funding rate history for an asset

        Args:
            coin: Asset name (e.g., "BTC")
            start_time: Start timestamp in milliseconds

        Returns:
            List of funding rate records
        """
        payload = {"type": "fundingHistory", "coin": coin, "startTime": start_time}
        return self._post_request(payload)

    def calculate_liquidation_price(
        self,
        entry_price: float,
        size: float,
        margin: float,
        leverage: float,
        coin: str,
    ) -> Optional[float]:
        """
        Calculate the liquidation price for a position

        Formula: liq_price = price - side * margin_available / position_size / (1 - l * side)
        where l = 1 / MAINTENANCE_LEVERAGE

        Args:
            entry_price: Position entry price
            size: Position size (positive for long, negative for short)
            margin: Available margin
            leverage: Current leverage
            coin: Asset name

        Returns:
            Calculated liquidation price or None if cannot be calculated
        """
        if size == 0 or margin == 0:
            return None

        side = 1 if size > 0 else -1
        position_size = abs(size)

        # Get max leverage for this asset
        max_leverage = ASSET_MAX_LEVERAGE.get(coin, 20)
        maintenance_leverage = max_leverage * 2  # Maintenance is half of initial

        l = 1 / maintenance_leverage

        try:
            denominator = 1 - l * side
            if denominator == 0:
                return None

            liq_price = entry_price - (side * margin / position_size / denominator)

            # Sanity checks
            if liq_price <= 0:
                return None
            if side == 1 and liq_price >= entry_price:  # Long liq should be below entry
                return None
            if side == -1 and liq_price <= entry_price:  # Short liq should be above entry
                return None

            return liq_price
        except Exception:
            return None


class HyperliquidWebSocket:
    """
    WebSocket client for real-time data from Hyperliquid

    Provides real-time updates for:
    - Trades
    - Order book
    - User events
    """

    def __init__(self, url: str = "wss://api.hyperliquid.xyz/ws"):
        self.url = url
        self.ws = None
        self.callbacks = {}

    async def connect(self):
        """Establish WebSocket connection"""
        import websockets

        self.ws = await websockets.connect(self.url)

    async def subscribe_trades(self, coin: str, callback):
        """Subscribe to trade updates for a coin"""
        if not self.ws:
            await self.connect()

        subscribe_msg = {
            "method": "subscribe",
            "subscription": {"type": "trades", "coin": coin},
        }
        await self.ws.send(json.dumps(subscribe_msg))
        self.callbacks[f"trades_{coin}"] = callback

    async def subscribe_l2_book(self, coin: str, callback):
        """Subscribe to L2 order book updates"""
        if not self.ws:
            await self.connect()

        subscribe_msg = {
            "method": "subscribe",
            "subscription": {"type": "l2Book", "coin": coin},
        }
        await self.ws.send(json.dumps(subscribe_msg))
        self.callbacks[f"l2Book_{coin}"] = callback

    async def subscribe_user_events(self, address: str, callback):
        """Subscribe to user events (fills, liquidations, etc.)"""
        if not self.ws:
            await self.connect()

        subscribe_msg = {
            "method": "subscribe",
            "subscription": {"type": "userEvents", "user": address},
        }
        await self.ws.send(json.dumps(subscribe_msg))
        self.callbacks[f"userEvents_{address}"] = callback

    async def listen(self):
        """Listen for incoming messages and dispatch to callbacks"""
        if not self.ws:
            await self.connect()

        async for message in self.ws:
            data = json.loads(message)
            channel = data.get("channel", "")

            for key, callback in self.callbacks.items():
                if key in channel:
                    await callback(data)

    async def close(self):
        """Close the WebSocket connection"""
        if self.ws:
            await self.ws.close()


def get_api_client(use_testnet: bool = False) -> HyperliquidAPI:
    """
    Factory function to create an API client instance

    Args:
        use_testnet: Whether to use testnet API

    Returns:
        Configured HyperliquidAPI instance
    """
    return HyperliquidAPI(
        use_testnet=use_testnet,
        cache_ttl=DEFAULT_LIQUIDATION_CONFIG.cache_ttl,
        rate_limit=DEFAULT_LIQUIDATION_CONFIG.rate_limit,
    )
