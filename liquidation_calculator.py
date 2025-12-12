"""
Liquidation Level Calculator for Hyperliquid

Calculates and aggregates liquidation levels across multiple positions
to create a comprehensive view of where liquidations are clustered.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

from config import (
    TRACKED_ASSETS,
    ASSET_MAX_LEVERAGE,
    DEFAULT_LIQUIDATION_CONFIG,
)
from hyperliquid_api import Position, MarketData


@dataclass
class LiquidationLevel:
    """Represents an aggregated liquidation level"""

    price: float
    total_long_size: float  # Total size of longs liquidated at or below this price
    total_short_size: float  # Total size of shorts liquidated at or above this price
    long_value_usd: float  # USD value of long liquidations
    short_value_usd: float  # USD value of short liquidations
    long_count: int  # Number of long positions
    short_count: int  # Number of short positions

    @property
    def total_value_usd(self) -> float:
        return self.long_value_usd + self.short_value_usd

    @property
    def net_value_usd(self) -> float:
        """Positive = more long liquidations, negative = more short liquidations"""
        return self.long_value_usd - self.short_value_usd


@dataclass
class LiquidationCluster:
    """Represents a cluster of liquidation levels"""

    price_low: float
    price_high: float
    center_price: float
    total_value_usd: float
    long_value_usd: float
    short_value_usd: float
    position_count: int
    density: float  # Value per price unit


@dataclass
class LiquidationHeatmapData:
    """Data structure for heatmap visualization"""

    coin: str
    current_price: float
    price_levels: np.ndarray
    long_values: np.ndarray  # USD value at each level (longs)
    short_values: np.ndarray  # USD value at each level (shorts)
    long_counts: np.ndarray  # Position count at each level (longs)
    short_counts: np.ndarray  # Position count at each level (shorts)
    timestamp: datetime
    total_positions_analyzed: int

    # Key levels
    major_long_levels: List[Tuple[float, float]]  # (price, value)
    major_short_levels: List[Tuple[float, float]]


class LiquidationCalculator:
    """
    Calculator for aggregating and analyzing liquidation levels

    Takes positions from various sources and computes:
    - Liquidation price distribution
    - Cumulative liquidation values
    - Key support/resistance levels based on liquidation clusters
    """

    def __init__(self, config: Optional[Any] = None):
        self.config = config or DEFAULT_LIQUIDATION_CONFIG

    def calculate_position_liquidation(
        self,
        entry_price: float,
        size: float,
        leverage: float,
        margin_type: str,
        coin: str,
        current_price: Optional[float] = None,
    ) -> Optional[float]:
        """
        Calculate liquidation price for a position

        Args:
            entry_price: Position entry price
            size: Position size (positive for long, negative for short)
            leverage: Position leverage
            margin_type: "cross" or "isolated"
            coin: Asset name
            current_price: Current market price (optional, for validation)

        Returns:
            Liquidation price or None if cannot be calculated
        """
        if size == 0 or entry_price <= 0:
            return None

        side = 1 if size > 0 else -1  # 1 for long, -1 for short

        # Get maintenance margin rate
        max_leverage = ASSET_MAX_LEVERAGE.get(coin, 20)
        # Maintenance margin is half of initial margin at max leverage
        # maintenance_rate = 1 / (2 * max_leverage)
        maintenance_rate = 0.5 / max_leverage

        # For isolated margin, liquidation depends on allocated margin
        # For cross margin, it depends on total account margin
        if margin_type == "isolated":
            # Initial margin = position_value / leverage
            # Position is liquidated when: margin <= maintenance_margin
            # maintenance_margin = position_value * maintenance_rate

            # Simplified formula for isolated:
            # Long: liq_price = entry_price * (1 - 1/leverage + maintenance_rate)
            # Short: liq_price = entry_price * (1 + 1/leverage - maintenance_rate)

            if side == 1:  # Long
                liq_price = entry_price * (1 - (1 / leverage) + maintenance_rate)
            else:  # Short
                liq_price = entry_price * (1 + (1 / leverage) - maintenance_rate)
        else:
            # Cross margin - more complex as it depends on total account state
            # Use approximate formula
            margin_ratio = 1 / leverage

            if side == 1:  # Long
                liq_price = entry_price * (1 - margin_ratio + maintenance_rate)
            else:  # Short
                liq_price = entry_price * (1 + margin_ratio - maintenance_rate)

        # Sanity checks
        if liq_price <= 0:
            return None

        # For longs, liq price should be below entry
        if side == 1 and liq_price >= entry_price:
            # Adjust using a safer formula
            liq_price = entry_price * (1 - 0.9 / leverage)

        # For shorts, liq price should be above entry
        if side == -1 and liq_price <= entry_price:
            liq_price = entry_price * (1 + 0.9 / leverage)

        return max(0.01, liq_price)  # Ensure positive price

    def aggregate_positions(
        self,
        positions: List[Position],
        coin: str,
        current_price: float,
        num_levels: Optional[int] = None,
        price_range_pct: Optional[float] = None,
    ) -> LiquidationHeatmapData:
        """
        Aggregate positions into liquidation level data for heatmap

        Args:
            positions: List of Position objects
            coin: Asset to filter for
            current_price: Current market price
            num_levels: Number of price levels (default from config)
            price_range_pct: Price range percentage (default from config)

        Returns:
            LiquidationHeatmapData object
        """
        num_levels = num_levels or self.config.price_levels
        price_range_pct = price_range_pct or self.config.price_range_percent

        # Filter positions for the specified coin
        coin_positions = [p for p in positions if p.coin == coin]

        # Calculate price range
        price_min = current_price * (1 - price_range_pct / 100)
        price_max = current_price * (1 + price_range_pct / 100)
        price_levels = np.linspace(price_min, price_max, num_levels)

        # Initialize arrays
        long_values = np.zeros(num_levels)
        short_values = np.zeros(num_levels)
        long_counts = np.zeros(num_levels, dtype=int)
        short_counts = np.zeros(num_levels, dtype=int)

        # Process each position
        for pos in coin_positions:
            # Get or calculate liquidation price
            liq_price = pos.liquidation_price
            if liq_price is None:
                liq_price = self.calculate_position_liquidation(
                    entry_price=pos.entry_price,
                    size=pos.size,
                    leverage=pos.leverage,
                    margin_type=pos.margin_type,
                    coin=coin,
                    current_price=current_price,
                )

            if liq_price is None:
                continue

            # Skip if outside price range
            if liq_price < price_min or liq_price > price_max:
                continue

            # Find the nearest price level
            level_idx = np.argmin(np.abs(price_levels - liq_price))

            # Calculate position value
            position_value = abs(pos.size) * pos.entry_price

            # Add to appropriate side
            if pos.is_long:
                # Long liquidations occur when price falls
                long_values[level_idx] += position_value
                long_counts[level_idx] += 1
            else:
                # Short liquidations occur when price rises
                short_values[level_idx] += position_value
                short_counts[level_idx] += 1

        # Find major liquidation levels
        major_long_levels = self._find_major_levels(
            price_levels, long_values, current_price, is_long=True
        )
        major_short_levels = self._find_major_levels(
            price_levels, short_values, current_price, is_long=False
        )

        return LiquidationHeatmapData(
            coin=coin,
            current_price=current_price,
            price_levels=price_levels,
            long_values=long_values,
            short_values=short_values,
            long_counts=long_counts,
            short_counts=short_counts,
            timestamp=datetime.now(),
            total_positions_analyzed=len(coin_positions),
            major_long_levels=major_long_levels,
            major_short_levels=major_short_levels,
        )

    def _find_major_levels(
        self,
        price_levels: np.ndarray,
        values: np.ndarray,
        current_price: float,
        is_long: bool,
        top_n: int = 5,
    ) -> List[Tuple[float, float]]:
        """
        Find the most significant liquidation levels

        Args:
            price_levels: Array of price levels
            values: Array of values at each level
            current_price: Current market price
            is_long: Whether these are long positions
            top_n: Number of top levels to return

        Returns:
            List of (price, value) tuples for major levels
        """
        # For longs, look below current price
        # For shorts, look above current price
        if is_long:
            mask = price_levels < current_price
        else:
            mask = price_levels > current_price

        filtered_prices = price_levels[mask]
        filtered_values = values[mask]

        if len(filtered_values) == 0:
            return []

        # Find indices of top values
        top_indices = np.argsort(filtered_values)[-top_n:][::-1]

        return [
            (float(filtered_prices[i]), float(filtered_values[i]))
            for i in top_indices
            if filtered_values[i] > 0
        ]

    def calculate_cumulative_liquidations(
        self,
        heatmap_data: LiquidationHeatmapData,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate cumulative liquidation values

        For longs: cumulative from current price downward
        For shorts: cumulative from current price upward

        Returns:
            Tuple of (cumulative_long, cumulative_short) arrays
        """
        price_levels = heatmap_data.price_levels
        current_price = heatmap_data.current_price

        # Find index closest to current price
        current_idx = np.argmin(np.abs(price_levels - current_price))

        # Cumulative longs: sum from current price going down
        cumulative_long = np.zeros_like(heatmap_data.long_values)
        for i in range(current_idx, -1, -1):
            if i == current_idx:
                cumulative_long[i] = heatmap_data.long_values[i]
            else:
                cumulative_long[i] = cumulative_long[i + 1] + heatmap_data.long_values[i]

        # Cumulative shorts: sum from current price going up
        cumulative_short = np.zeros_like(heatmap_data.short_values)
        for i in range(current_idx, len(price_levels)):
            if i == current_idx:
                cumulative_short[i] = heatmap_data.short_values[i]
            else:
                cumulative_short[i] = cumulative_short[i - 1] + heatmap_data.short_values[i]

        return cumulative_long, cumulative_short

    def find_liquidation_clusters(
        self,
        heatmap_data: LiquidationHeatmapData,
        min_cluster_value: float = 1000000,  # $1M minimum
        cluster_width_pct: float = 1.0,  # 1% price width
    ) -> List[LiquidationCluster]:
        """
        Find clusters of liquidations (areas with high concentration)

        Args:
            heatmap_data: Heatmap data to analyze
            min_cluster_value: Minimum USD value for a cluster
            cluster_width_pct: Width of cluster as % of price

        Returns:
            List of LiquidationCluster objects
        """
        clusters = []
        price_levels = heatmap_data.price_levels
        combined_values = heatmap_data.long_values + heatmap_data.short_values

        # Calculate cluster width in price units
        cluster_width = heatmap_data.current_price * (cluster_width_pct / 100)
        levels_per_cluster = max(1, int(cluster_width / (price_levels[1] - price_levels[0])))

        i = 0
        while i < len(price_levels) - levels_per_cluster:
            # Sum values in this window
            window_long = np.sum(heatmap_data.long_values[i : i + levels_per_cluster])
            window_short = np.sum(heatmap_data.short_values[i : i + levels_per_cluster])
            window_total = window_long + window_short

            if window_total >= min_cluster_value:
                # Count positions in window
                pos_count = int(
                    np.sum(heatmap_data.long_counts[i : i + levels_per_cluster])
                    + np.sum(heatmap_data.short_counts[i : i + levels_per_cluster])
                )

                cluster = LiquidationCluster(
                    price_low=price_levels[i],
                    price_high=price_levels[i + levels_per_cluster - 1],
                    center_price=(price_levels[i] + price_levels[i + levels_per_cluster - 1])
                    / 2,
                    total_value_usd=window_total,
                    long_value_usd=window_long,
                    short_value_usd=window_short,
                    position_count=pos_count,
                    density=window_total / cluster_width,
                )
                clusters.append(cluster)

                # Skip past this cluster
                i += levels_per_cluster
            else:
                i += 1

        # Sort by total value descending
        clusters.sort(key=lambda c: c.total_value_usd, reverse=True)

        return clusters

    def generate_summary_stats(
        self,
        heatmap_data: LiquidationHeatmapData,
    ) -> Dict[str, Any]:
        """
        Generate summary statistics for the liquidation data

        Returns:
            Dictionary of statistics
        """
        total_long_value = float(np.sum(heatmap_data.long_values))
        total_short_value = float(np.sum(heatmap_data.short_values))
        total_long_count = int(np.sum(heatmap_data.long_counts))
        total_short_count = int(np.sum(heatmap_data.short_counts))

        # Find nearest major levels
        nearest_long = None
        for price, value in heatmap_data.major_long_levels:
            if price < heatmap_data.current_price:
                nearest_long = (price, value)
                break

        nearest_short = None
        for price, value in heatmap_data.major_short_levels:
            if price > heatmap_data.current_price:
                nearest_short = (price, value)
                break

        return {
            "coin": heatmap_data.coin,
            "current_price": heatmap_data.current_price,
            "timestamp": heatmap_data.timestamp.isoformat(),
            "total_positions": heatmap_data.total_positions_analyzed,
            "long_positions": total_long_count,
            "short_positions": total_short_count,
            "total_long_value_usd": total_long_value,
            "total_short_value_usd": total_short_value,
            "long_short_ratio": (
                total_long_value / total_short_value if total_short_value > 0 else float("inf")
            ),
            "nearest_long_liquidation": nearest_long,
            "nearest_short_liquidation": nearest_short,
            "major_long_levels": heatmap_data.major_long_levels,
            "major_short_levels": heatmap_data.major_short_levels,
        }


class PositionAggregator:
    """
    Aggregates positions from multiple data sources
    """

    def __init__(self):
        self.positions: Dict[str, List[Position]] = defaultdict(list)
        self.last_update: Dict[str, datetime] = {}

    def add_positions(
        self,
        positions: List[Position],
        source: str = "unknown",
    ):
        """Add positions from a data source"""
        for pos in positions:
            self.positions[pos.coin].append(pos)
        self.last_update[source] = datetime.now()

    def get_positions_for_coin(self, coin: str) -> List[Position]:
        """Get all positions for a specific coin"""
        return self.positions.get(coin, [])

    def get_all_positions(self) -> List[Position]:
        """Get all positions across all coins"""
        all_positions = []
        for coin_positions in self.positions.values():
            all_positions.extend(coin_positions)
        return all_positions

    def clear(self):
        """Clear all stored positions"""
        self.positions.clear()
        self.last_update.clear()

    def deduplicate(self):
        """Remove duplicate positions (same address and coin)"""
        for coin in self.positions:
            seen = set()
            unique = []
            for pos in self.positions[coin]:
                key = (pos.address, pos.coin, pos.size)
                if key not in seen:
                    seen.add(key)
                    unique.append(pos)
            self.positions[coin] = unique


def create_sample_positions(
    coin: str,
    current_price: float,
    count: int = 100,
) -> List[Position]:
    """
    Create sample positions for testing/demo purposes

    Args:
        coin: Asset name
        current_price: Current market price
        count: Number of positions to generate

    Returns:
        List of sample Position objects
    """
    positions = []
    np.random.seed(42)  # For reproducibility

    for i in range(count):
        # Random side
        is_long = np.random.random() > 0.5

        # Random leverage (1-20x)
        leverage = np.random.randint(1, 21)

        # Random entry price (within 10% of current)
        entry_deviation = np.random.uniform(-0.10, 0.10)
        entry_price = current_price * (1 + entry_deviation)

        # Random size (in USD value)
        position_value = np.random.exponential(100000)  # Average $100k
        size = position_value / entry_price
        if not is_long:
            size = -size

        # Calculate approximate liquidation price
        max_leverage = ASSET_MAX_LEVERAGE.get(coin, 20)
        maintenance_rate = 0.5 / max_leverage

        if is_long:
            liq_price = entry_price * (1 - (1 / leverage) + maintenance_rate)
        else:
            liq_price = entry_price * (1 + (1 / leverage) - maintenance_rate)

        position = Position(
            address=f"0x{i:040x}",
            coin=coin,
            size=size,
            entry_price=entry_price,
            mark_price=current_price,
            liquidation_price=liq_price,
            leverage=leverage,
            margin_used=position_value / leverage,
            position_value=position_value,
            unrealized_pnl=(current_price - entry_price) * size,
            margin_type="isolated" if np.random.random() > 0.7 else "cross",
            timestamp=datetime.now(),
        )
        positions.append(position)

    return positions
