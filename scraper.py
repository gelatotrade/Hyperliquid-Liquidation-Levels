"""
Web scraper for Hypurrscan.io to fetch whale trades and large positions
"""

import asyncio
import re
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin

import aiohttp
import requests
from bs4 import BeautifulSoup

from config import (
    HYPURRSCAN_BASE_URL,
    DEFAULT_SCRAPING_CONFIG,
    TRACKED_ASSETS,
)


@dataclass
class WhaleTrade:
    """Represents a large trade from a whale"""

    address: str
    coin: str
    side: str  # "buy" or "sell"
    size: float
    price: float
    value_usd: float
    timestamp: datetime
    tx_hash: Optional[str] = None


@dataclass
class WhalePosition:
    """Represents a whale's position scraped from Hypurrscan"""

    address: str
    coin: str
    side: str  # "long" or "short"
    size: float
    entry_price: float
    mark_price: float
    liquidation_price: Optional[float]
    pnl: float
    pnl_percent: float
    value_usd: float
    leverage: float
    last_updated: datetime


class HypurrscanScraper:
    """
    Scraper for Hypurrscan.io to fetch whale data

    Note: This scraper is for educational purposes.
    Hypurrscan may have rate limits or terms of service regarding automated access.
    Always respect the website's robots.txt and terms of service.
    """

    def __init__(self, config: Optional[Any] = None):
        self.config = config or DEFAULT_SCRAPING_CONFIG
        self.base_url = HYPURRSCAN_BASE_URL
        self.session: Optional[aiohttp.ClientSession] = None
        self.last_request_time = 0.0

        self.headers = {
            "User-Agent": self.config.user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
        }

    def _rate_limit_wait(self):
        """Enforce rate limiting between requests"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.config.request_delay:
            time.sleep(self.config.request_delay - elapsed)
        self.last_request_time = time.time()

    def _make_request(
        self, url: str, retry_count: int = 0
    ) -> Optional[requests.Response]:
        """Make an HTTP request with retry logic"""
        try:
            self._rate_limit_wait()
            response = requests.get(url, headers=self.headers, timeout=30)
            response.raise_for_status()
            return response
        except requests.RequestException as e:
            if retry_count < self.config.max_retries:
                time.sleep(self.config.retry_delay * (retry_count + 1))
                return self._make_request(url, retry_count + 1)
            print(f"Request failed after {self.config.max_retries} retries: {e}")
            return None

    def get_whale_leaderboard(self) -> List[Dict[str, Any]]:
        """
        Fetch the whale leaderboard from Hypurrscan

        Returns:
            List of whale entries with address and stats
        """
        url = urljoin(self.base_url, "/leaderboard")
        response = self._make_request(url)

        if not response:
            return []

        whales = []
        soup = BeautifulSoup(response.text, "lxml")

        # Look for leaderboard table or data elements
        # Note: Actual selectors depend on the website's HTML structure
        # These are placeholder patterns that may need adjustment

        # Try to find trader cards or table rows
        trader_elements = soup.select(
            ".trader-card, .leaderboard-row, [data-trader], tr[data-address]"
        )

        for element in trader_elements:
            try:
                # Extract address
                address = (
                    element.get("data-address")
                    or element.get("data-trader")
                    or self._extract_address(element)
                )

                if not address:
                    continue

                # Extract PnL and volume info
                pnl_element = element.select_one(".pnl, .profit, [data-pnl]")
                volume_element = element.select_one(".volume, [data-volume]")

                whale = {
                    "address": address,
                    "pnl": self._parse_number(pnl_element.text if pnl_element else "0"),
                    "volume": self._parse_number(
                        volume_element.text if volume_element else "0"
                    ),
                }
                whales.append(whale)
            except Exception as e:
                print(f"Error parsing whale entry: {e}")
                continue

        return whales

    def get_recent_large_trades(
        self,
        coin: Optional[str] = None,
        min_value_usd: float = 100000,
    ) -> List[WhaleTrade]:
        """
        Fetch recent large trades from Hypurrscan

        Args:
            coin: Optional asset filter (e.g., "BTC")
            min_value_usd: Minimum trade value in USD

        Returns:
            List of WhaleTrade objects
        """
        url = urljoin(self.base_url, "/trades")
        if coin:
            url = f"{url}?coin={coin}"

        response = self._make_request(url)
        if not response:
            return []

        trades = []
        soup = BeautifulSoup(response.text, "lxml")

        # Find trade entries
        trade_elements = soup.select(
            ".trade-row, .trade-entry, [data-trade], tr.trade"
        )

        for element in trade_elements:
            try:
                trade = self._parse_trade_element(element)
                if trade and trade.value_usd >= min_value_usd:
                    if coin is None or trade.coin == coin:
                        trades.append(trade)
            except Exception as e:
                print(f"Error parsing trade: {e}")
                continue

        return trades

    def get_whale_positions(
        self,
        address: str,
    ) -> List[WhalePosition]:
        """
        Fetch positions for a specific whale address from Hypurrscan

        Args:
            address: Wallet address to query

        Returns:
            List of WhalePosition objects
        """
        url = urljoin(self.base_url, f"/address/{address}")
        response = self._make_request(url)

        if not response:
            return []

        positions = []
        soup = BeautifulSoup(response.text, "lxml")

        # Find position elements
        position_elements = soup.select(
            ".position-row, .position-card, [data-position], tr.position"
        )

        for element in position_elements:
            try:
                position = self._parse_position_element(element, address)
                if position:
                    positions.append(position)
            except Exception as e:
                print(f"Error parsing position: {e}")
                continue

        return positions

    def get_liquidations(
        self,
        coin: Optional[str] = None,
        hours: int = 24,
    ) -> List[Dict[str, Any]]:
        """
        Fetch recent liquidations from Hypurrscan

        Args:
            coin: Optional asset filter
            hours: Look back period in hours

        Returns:
            List of liquidation events
        """
        url = urljoin(self.base_url, "/liquidations")
        response = self._make_request(url)

        if not response:
            return []

        liquidations = []
        soup = BeautifulSoup(response.text, "lxml")

        # Find liquidation entries
        liq_elements = soup.select(
            ".liquidation-row, .liq-entry, [data-liquidation]"
        )

        for element in liq_elements:
            try:
                liq = self._parse_liquidation_element(element)
                if liq:
                    if coin is None or liq.get("coin") == coin:
                        liquidations.append(liq)
            except Exception as e:
                print(f"Error parsing liquidation: {e}")
                continue

        return liquidations

    def _extract_address(self, element) -> Optional[str]:
        """Extract wallet address from element"""
        # Look for address in various formats
        address_patterns = [
            r"0x[a-fA-F0-9]{40}",  # Standard Ethereum address
        ]

        text = element.get_text()
        for pattern in address_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(0)

        # Check href attributes
        links = element.select("a[href*='address'], a[href*='0x']")
        for link in links:
            href = link.get("href", "")
            match = re.search(r"0x[a-fA-F0-9]{40}", href)
            if match:
                return match.group(0)

        return None

    def _parse_number(self, text: str) -> float:
        """Parse number from text with various formats"""
        if not text:
            return 0.0

        # Remove common formatting
        text = text.strip()
        text = text.replace(",", "")
        text = text.replace("$", "")
        text = text.replace("%", "")

        # Handle K, M, B suffixes
        multipliers = {"K": 1000, "M": 1000000, "B": 1000000000}
        for suffix, mult in multipliers.items():
            if text.upper().endswith(suffix):
                try:
                    return float(text[:-1]) * mult
                except ValueError:
                    pass

        # Handle negative values in parentheses
        if text.startswith("(") and text.endswith(")"):
            text = "-" + text[1:-1]

        try:
            return float(text)
        except ValueError:
            return 0.0

    def _parse_trade_element(self, element) -> Optional[WhaleTrade]:
        """Parse a trade from HTML element"""
        try:
            address = self._extract_address(element)
            if not address:
                return None

            # Extract coin/asset
            coin_elem = element.select_one(".coin, .asset, [data-coin]")
            coin = coin_elem.text.strip() if coin_elem else "UNKNOWN"

            # Extract side
            side_elem = element.select_one(".side, .direction, [data-side]")
            side_text = side_elem.text.strip().lower() if side_elem else ""
            side = "buy" if "buy" in side_text or "long" in side_text else "sell"

            # Extract size
            size_elem = element.select_one(".size, .amount, [data-size]")
            size = self._parse_number(size_elem.text if size_elem else "0")

            # Extract price
            price_elem = element.select_one(".price, [data-price]")
            price = self._parse_number(price_elem.text if price_elem else "0")

            # Extract value
            value_elem = element.select_one(".value, .notional, [data-value]")
            value_usd = self._parse_number(value_elem.text if value_elem else "0")
            if value_usd == 0 and size > 0 and price > 0:
                value_usd = size * price

            # Extract timestamp
            time_elem = element.select_one(".time, .timestamp, [data-time]")
            timestamp = datetime.now()  # Default to now if not found

            return WhaleTrade(
                address=address,
                coin=coin,
                side=side,
                size=size,
                price=price,
                value_usd=value_usd,
                timestamp=timestamp,
            )
        except Exception:
            return None

    def _parse_position_element(
        self, element, address: str
    ) -> Optional[WhalePosition]:
        """Parse a position from HTML element"""
        try:
            # Extract coin/asset
            coin_elem = element.select_one(".coin, .asset, [data-coin]")
            coin = coin_elem.text.strip() if coin_elem else "UNKNOWN"

            # Extract side/direction
            side_elem = element.select_one(".side, .direction, [data-side]")
            side_text = side_elem.text.strip().lower() if side_elem else ""
            side = "long" if "long" in side_text or "buy" in side_text else "short"

            # Extract size
            size_elem = element.select_one(".size, .amount, [data-size]")
            size = self._parse_number(size_elem.text if size_elem else "0")

            # Extract prices
            entry_elem = element.select_one(".entry-price, [data-entry]")
            entry_price = self._parse_number(entry_elem.text if entry_elem else "0")

            mark_elem = element.select_one(".mark-price, [data-mark]")
            mark_price = self._parse_number(mark_elem.text if mark_elem else "0")

            liq_elem = element.select_one(".liq-price, .liquidation, [data-liq]")
            liq_price = self._parse_number(liq_elem.text if liq_elem else "0")
            liq_price = liq_price if liq_price > 0 else None

            # Extract PnL
            pnl_elem = element.select_one(".pnl, .profit, [data-pnl]")
            pnl = self._parse_number(pnl_elem.text if pnl_elem else "0")

            pnl_pct_elem = element.select_one(".pnl-percent, [data-pnl-pct]")
            pnl_percent = self._parse_number(pnl_pct_elem.text if pnl_pct_elem else "0")

            # Extract value and leverage
            value_elem = element.select_one(".value, .notional, [data-value]")
            value_usd = self._parse_number(value_elem.text if value_elem else "0")

            leverage_elem = element.select_one(".leverage, [data-leverage]")
            leverage = self._parse_number(leverage_elem.text if leverage_elem else "1")
            leverage = max(1, leverage)

            return WhalePosition(
                address=address,
                coin=coin,
                side=side,
                size=size,
                entry_price=entry_price,
                mark_price=mark_price,
                liquidation_price=liq_price,
                pnl=pnl,
                pnl_percent=pnl_percent,
                value_usd=value_usd,
                leverage=leverage,
                last_updated=datetime.now(),
            )
        except Exception:
            return None

    def _parse_liquidation_element(self, element) -> Optional[Dict[str, Any]]:
        """Parse a liquidation event from HTML element"""
        try:
            address = self._extract_address(element)

            coin_elem = element.select_one(".coin, .asset, [data-coin]")
            coin = coin_elem.text.strip() if coin_elem else "UNKNOWN"

            side_elem = element.select_one(".side, .direction, [data-side]")
            side_text = side_elem.text.strip().lower() if side_elem else ""
            side = "long" if "long" in side_text else "short"

            size_elem = element.select_one(".size, [data-size]")
            size = self._parse_number(size_elem.text if size_elem else "0")

            price_elem = element.select_one(".price, [data-price]")
            price = self._parse_number(price_elem.text if price_elem else "0")

            value_elem = element.select_one(".value, [data-value]")
            value = self._parse_number(value_elem.text if value_elem else "0")

            return {
                "address": address,
                "coin": coin,
                "side": side,
                "size": size,
                "price": price,
                "value_usd": value,
                "timestamp": datetime.now(),
            }
        except Exception:
            return None


class AlternativeDataSources:
    """
    Alternative data sources for whale tracking when scraping is not available

    Uses public APIs and blockchain data to find large positions
    """

    def __init__(self):
        self.hyperliquid_api_url = "https://api.hyperliquid.xyz/info"

    def get_top_traders_from_leaderboard(self) -> List[str]:
        """
        Fetch top trader addresses from Hyperliquid's public leaderboard data

        Returns:
            List of wallet addresses
        """
        try:
            # The leaderboard data might be available through the API
            # This is an approximation - actual endpoint may vary
            payload = {"type": "leaderboard"}
            response = requests.post(
                self.hyperliquid_api_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30,
            )

            if response.status_code == 200:
                data = response.json()
                # Extract addresses from leaderboard data
                addresses = []
                if isinstance(data, list):
                    for entry in data:
                        if isinstance(entry, dict) and "user" in entry:
                            addresses.append(entry["user"])
                        elif isinstance(entry, str) and entry.startswith("0x"):
                            addresses.append(entry)
                return addresses[:100]  # Top 100 traders
        except Exception as e:
            print(f"Error fetching leaderboard: {e}")

        return []

    def find_whale_addresses_from_trades(
        self,
        coin: str,
        min_size: float = 1.0,
        lookback_hours: int = 24,
    ) -> List[str]:
        """
        Find whale addresses by analyzing recent large trades

        Args:
            coin: Asset to analyze
            min_size: Minimum position size in asset units
            lookback_hours: How far back to look

        Returns:
            List of addresses with large trades
        """
        # This would require access to trade history
        # For now, return empty list as this needs WebSocket or historical data access
        return []


def get_scraper() -> HypurrscanScraper:
    """Factory function to create a scraper instance"""
    return HypurrscanScraper(config=DEFAULT_SCRAPING_CONFIG)


def get_alternative_sources() -> AlternativeDataSources:
    """Factory function to create alternative data source instance"""
    return AlternativeDataSources()
