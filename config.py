"""
Configuration settings for Hyperliquid Liquidation Heatmap
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional

# Hyperliquid API Endpoints
HYPERLIQUID_API_URL = "https://api.hyperliquid.xyz/info"
HYPERLIQUID_WS_URL = "wss://api.hyperliquid.xyz/ws"
HYPERLIQUID_TESTNET_API_URL = "https://api.hyperliquid-testnet.xyz/info"

# Hypurrscan base URL
HYPURRSCAN_BASE_URL = "https://hypurrscan.io"

# Assets to track
TRACKED_ASSETS = ["BTC", "ETH", "HYPE"]

# Leverage and margin settings per asset (max leverage varies by asset)
ASSET_MAX_LEVERAGE = {
    "BTC": 50,
    "ETH": 50,
    "HYPE": 20,
}

# Maintenance margin = 1 / (2 * max_leverage) for calculating liquidation
# This is approximate - actual values come from the exchange
MAINTENANCE_LEVERAGE_MULTIPLIER = 2


@dataclass
class LiquidationConfig:
    """Configuration for liquidation level calculations"""

    # Minimum position size in USD to consider
    min_position_usd: float = 10000.0

    # Whale threshold in USD
    whale_threshold_usd: float = 100000.0

    # Large position threshold in USD
    large_position_threshold_usd: float = 500000.0

    # Number of price levels for heatmap
    price_levels: int = 100

    # Price range percentage around current price
    price_range_percent: float = 20.0

    # Cache TTL in seconds
    cache_ttl: int = 60

    # Request timeout
    request_timeout: int = 30

    # Rate limiting (requests per second)
    rate_limit: float = 5.0

    # Number of top whale addresses to track
    top_whales_count: int = 100


@dataclass
class HeatmapConfig:
    """Configuration for heatmap visualization"""

    # Figure size
    figure_width: int = 16
    figure_height: int = 10

    # Color scheme
    long_color: str = "Greens"
    short_color: str = "Reds"
    combined_color: str = "RdYlGn"

    # Price axis settings
    show_current_price_line: bool = True
    current_price_color: str = "#FFD700"

    # Grid settings
    show_grid: bool = True
    grid_alpha: float = 0.3

    # Output settings
    output_dir: str = "output"
    output_format: str = "png"
    dpi: int = 150


@dataclass
class ScrapingConfig:
    """Configuration for web scraping"""

    # User agent for requests
    user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"

    # Retry settings
    max_retries: int = 3
    retry_delay: float = 2.0

    # Use headless browser for JavaScript-rendered content
    use_selenium: bool = False
    headless: bool = True

    # Request delay between scrapes (seconds)
    request_delay: float = 1.0


# Known whale addresses (frequently updated - these are examples)
# These addresses are publicly known large traders on Hyperliquid
KNOWN_WHALE_ADDRESSES: List[str] = [
    # Add known whale addresses here
    # Format: "0x..." (42-character hex addresses)
]

# Default configuration instances
DEFAULT_LIQUIDATION_CONFIG = LiquidationConfig()
DEFAULT_HEATMAP_CONFIG = HeatmapConfig()
DEFAULT_SCRAPING_CONFIG = ScrapingConfig()


def get_env_config() -> Dict[str, str]:
    """Load configuration from environment variables"""
    return {
        "api_url": os.getenv("HYPERLIQUID_API_URL", HYPERLIQUID_API_URL),
        "use_testnet": os.getenv("USE_TESTNET", "false").lower() == "true",
        "output_dir": os.getenv("OUTPUT_DIR", "output"),
        "cache_ttl": int(os.getenv("CACHE_TTL", "60")),
    }
