#!/usr/bin/env python3
"""
Hyperliquid Liquidation Heatmap

Main entry point for generating liquidation level heatmaps for BTC, ETH, and HYPE
on the Hyperliquid decentralized exchange.

Features:
- Fetches position data from Hyperliquid API
- Scrapes whale trades and positions from Hypurrscan.io
- Calculates liquidation levels for tracked positions
- Generates visual heatmaps showing liquidation clusters
- Supports both static (PNG) and interactive (HTML) visualizations

Usage:
    python main.py                     # Run with default settings
    python main.py --demo              # Run with sample data (no API calls)
    python main.py --assets BTC ETH    # Specify assets to track
    python main.py --interactive       # Generate interactive HTML charts
    python main.py --continuous        # Run continuously with updates

Author: Generated with Claude Code
"""

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from config import (
    TRACKED_ASSETS,
    DEFAULT_LIQUIDATION_CONFIG,
    DEFAULT_HEATMAP_CONFIG,
    KNOWN_WHALE_ADDRESSES,
)
from hyperliquid_api import HyperliquidAPI, Position, get_api_client
from scraper import HypurrscanScraper, AlternativeDataSources, get_scraper
from liquidation_calculator import (
    LiquidationCalculator,
    LiquidationHeatmapData,
    PositionAggregator,
    create_sample_positions,
)
from heatmap_visualizer import (
    HeatmapVisualizer,
    TerminalVisualizer,
    get_visualizer,
    get_terminal_visualizer,
)


class LiquidationHeatmapGenerator:
    """
    Main class for generating liquidation heatmaps

    Coordinates data fetching, calculation, and visualization
    """

    def __init__(
        self,
        assets: Optional[List[str]] = None,
        use_testnet: bool = False,
        output_dir: str = "output",
    ):
        """
        Initialize the heatmap generator

        Args:
            assets: List of assets to track (default: BTC, ETH, HYPE)
            use_testnet: Whether to use Hyperliquid testnet
            output_dir: Directory for output files
        """
        self.assets = assets or TRACKED_ASSETS
        self.use_testnet = use_testnet
        self.output_dir = output_dir

        # Initialize components
        self.api = get_api_client(use_testnet=use_testnet)
        self.scraper = get_scraper()
        self.alternative_sources = AlternativeDataSources()
        self.calculator = LiquidationCalculator()
        self.visualizer = get_visualizer()
        self.terminal_viz = get_terminal_visualizer()

        # Position aggregator
        self.aggregator = PositionAggregator()

        # Ensure output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Update visualizer config
        self.visualizer.config.output_dir = output_dir

    def fetch_market_data(self) -> Dict[str, float]:
        """
        Fetch current market prices for tracked assets

        Returns:
            Dictionary mapping coin to current price
        """
        print("Fetching market data from Hyperliquid...")

        try:
            assets_info, market_data = self.api.get_meta_and_asset_contexts()
            prices = {}

            for coin in self.assets:
                if coin in market_data:
                    prices[coin] = market_data[coin].mark_price
                    print(f"  {coin}: ${prices[coin]:,.2f}")
                else:
                    print(f"  {coin}: Price not found, skipping")

            return prices

        except Exception as e:
            print(f"Error fetching market data: {e}")
            # Return fallback prices for demo
            return {"BTC": 95000.0, "ETH": 3500.0, "HYPE": 25.0}

    def fetch_whale_addresses(self) -> List[str]:
        """
        Fetch list of whale addresses to track

        Returns:
            List of wallet addresses
        """
        print("Fetching whale addresses...")
        addresses = []

        # Start with known whale addresses
        addresses.extend(KNOWN_WHALE_ADDRESSES)

        # Try to fetch from leaderboard
        try:
            leaderboard_addresses = self.alternative_sources.get_top_traders_from_leaderboard()
            addresses.extend(leaderboard_addresses)
            print(f"  Found {len(leaderboard_addresses)} addresses from leaderboard")
        except Exception as e:
            print(f"  Could not fetch leaderboard: {e}")

        # Try to scrape from Hypurrscan
        try:
            whales = self.scraper.get_whale_leaderboard()
            for whale in whales:
                if "address" in whale:
                    addresses.append(whale["address"])
            print(f"  Found {len(whales)} addresses from Hypurrscan")
        except Exception as e:
            print(f"  Could not scrape Hypurrscan: {e}")

        # Deduplicate
        addresses = list(set(addresses))
        print(f"  Total unique addresses: {len(addresses)}")

        return addresses

    def fetch_positions(
        self,
        addresses: List[str],
        max_addresses: int = 100,
    ) -> List[Position]:
        """
        Fetch positions for whale addresses

        Args:
            addresses: List of wallet addresses
            max_addresses: Maximum number of addresses to query

        Returns:
            List of Position objects
        """
        print(f"Fetching positions for up to {max_addresses} addresses...")
        all_positions = []

        # Limit addresses
        addresses = addresses[:max_addresses]

        for i, address in enumerate(addresses):
            try:
                positions = self.api.get_user_positions(address)
                all_positions.extend(positions)

                if (i + 1) % 10 == 0:
                    print(f"  Processed {i + 1}/{len(addresses)} addresses...")

            except Exception as e:
                # Skip failed addresses
                continue

        print(f"  Total positions fetched: {len(all_positions)}")
        return all_positions

    def fetch_large_trades(self) -> List[Dict[str, Any]]:
        """
        Fetch recent large trades from Hypurrscan

        Returns:
            List of trade data
        """
        print("Fetching large trades from Hypurrscan...")
        all_trades = []

        for coin in self.assets:
            try:
                trades = self.scraper.get_recent_large_trades(
                    coin=coin,
                    min_value_usd=DEFAULT_LIQUIDATION_CONFIG.whale_threshold_usd,
                )
                all_trades.extend(trades)
                print(f"  {coin}: {len(trades)} large trades found")
            except Exception as e:
                print(f"  {coin}: Could not fetch trades ({e})")

        return all_trades

    def generate_heatmaps(
        self,
        positions: List[Position],
        prices: Dict[str, float],
        interactive: bool = False,
    ) -> List[str]:
        """
        Generate heatmaps for all tracked assets

        Args:
            positions: List of positions to analyze
            prices: Current prices for each asset
            interactive: Whether to generate interactive HTML charts

        Returns:
            List of output file paths
        """
        output_files = []

        for coin in self.assets:
            if coin not in prices:
                print(f"Skipping {coin}: no price data")
                continue

            current_price = prices[coin]
            print(f"\nGenerating heatmap for {coin}...")

            # Filter positions for this coin
            coin_positions = [p for p in positions if p.coin == coin]
            print(f"  Positions: {len(coin_positions)}")

            if len(coin_positions) == 0:
                print(f"  No positions found for {coin}, skipping")
                continue

            # Calculate heatmap data
            heatmap_data = self.calculator.aggregate_positions(
                positions=coin_positions,
                coin=coin,
                current_price=current_price,
            )

            # Generate summary stats
            stats = self.calculator.generate_summary_stats(heatmap_data)

            # Print terminal summary
            self.terminal_viz.print_summary(heatmap_data, stats)

            # Generate visualizations
            # Combined heatmap (static)
            filepath = self.visualizer.create_combined_heatmap(
                data=heatmap_data,
                save=True,
                show=False,
            )
            if filepath:
                output_files.append(filepath)

            # Interactive heatmap (if requested)
            if interactive:
                filepath = self.visualizer.create_interactive_heatmap(
                    data=heatmap_data,
                    save=True,
                )
                if filepath:
                    output_files.append(filepath)

            # Save stats to JSON
            stats_file = os.path.join(
                self.output_dir,
                f"{coin}_liquidation_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            )
            with open(stats_file, "w") as f:
                # Convert non-serializable types
                serializable_stats = {
                    k: (v if not isinstance(v, datetime) else v.isoformat())
                    for k, v in stats.items()
                }
                json.dump(serializable_stats, f, indent=2, default=str)
            output_files.append(stats_file)

        return output_files

    def run_demo(self, interactive: bool = False) -> List[str]:
        """
        Run with sample data for demonstration

        Args:
            interactive: Whether to generate interactive charts

        Returns:
            List of output file paths
        """
        print("\n" + "=" * 60)
        print("RUNNING IN DEMO MODE (using sample data)")
        print("=" * 60 + "\n")

        # Sample prices
        prices = {
            "BTC": 95000.0,
            "ETH": 3500.0,
            "HYPE": 25.0,
        }

        # Generate sample positions for each asset
        all_positions = []
        for coin in self.assets:
            if coin in prices:
                sample = create_sample_positions(
                    coin=coin,
                    current_price=prices[coin],
                    count=200,  # 200 sample positions per asset
                )
                all_positions.extend(sample)
                print(f"Generated {len(sample)} sample positions for {coin}")

        # Generate heatmaps
        return self.generate_heatmaps(all_positions, prices, interactive)

    def run(
        self,
        demo: bool = False,
        interactive: bool = False,
        max_whale_addresses: int = 50,
    ) -> List[str]:
        """
        Run the full pipeline

        Args:
            demo: Use sample data instead of fetching real data
            interactive: Generate interactive HTML charts
            max_whale_addresses: Maximum whale addresses to query

        Returns:
            List of output file paths
        """
        if demo:
            return self.run_demo(interactive)

        print("\n" + "=" * 60)
        print("HYPERLIQUID LIQUIDATION HEATMAP GENERATOR")
        print("=" * 60 + "\n")

        # Step 1: Fetch market data
        prices = self.fetch_market_data()

        if not prices:
            print("Error: Could not fetch market data")
            return []

        # Step 2: Fetch whale addresses
        addresses = self.fetch_whale_addresses()

        # If no addresses found, use demo mode
        if not addresses:
            print("\nNo whale addresses found, switching to demo mode...")
            return self.run_demo(interactive)

        # Step 3: Fetch positions
        positions = self.fetch_positions(addresses, max_whale_addresses)

        if not positions:
            print("\nNo positions found, switching to demo mode...")
            return self.run_demo(interactive)

        # Step 4: Fetch large trades (optional, for additional context)
        try:
            self.fetch_large_trades()
        except Exception as e:
            print(f"Could not fetch large trades: {e}")

        # Step 5: Generate heatmaps
        return self.generate_heatmaps(positions, prices, interactive)

    def run_continuous(
        self,
        interval_seconds: int = 300,
        interactive: bool = False,
    ):
        """
        Run continuously, updating heatmaps at regular intervals

        Args:
            interval_seconds: Update interval in seconds (default: 5 minutes)
            interactive: Generate interactive charts
        """
        print(f"\nStarting continuous mode (updating every {interval_seconds}s)")
        print("Press Ctrl+C to stop\n")

        try:
            while True:
                try:
                    output_files = self.run(interactive=interactive)
                    print(f"\nGenerated {len(output_files)} files")
                    print(f"Next update in {interval_seconds} seconds...")
                except Exception as e:
                    print(f"Error during update: {e}")

                time.sleep(interval_seconds)

        except KeyboardInterrupt:
            print("\n\nStopped by user")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Generate liquidation level heatmaps for Hyperliquid",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                        # Run with real data
  python main.py --demo                 # Run with sample data
  python main.py --assets BTC ETH       # Track specific assets
  python main.py --interactive          # Generate HTML charts
  python main.py --continuous --interval 60  # Update every minute
        """,
    )

    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run in demo mode with sample data",
    )

    parser.add_argument(
        "--assets",
        nargs="+",
        default=TRACKED_ASSETS,
        help=f"Assets to track (default: {', '.join(TRACKED_ASSETS)})",
    )

    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Generate interactive HTML charts (requires plotly)",
    )

    parser.add_argument(
        "--continuous",
        "-c",
        action="store_true",
        help="Run continuously with periodic updates",
    )

    parser.add_argument(
        "--interval",
        type=int,
        default=300,
        help="Update interval in seconds for continuous mode (default: 300)",
    )

    parser.add_argument(
        "--output",
        "-o",
        default="output",
        help="Output directory for generated files (default: output)",
    )

    parser.add_argument(
        "--max-whales",
        type=int,
        default=50,
        help="Maximum number of whale addresses to query (default: 50)",
    )

    parser.add_argument(
        "--testnet",
        action="store_true",
        help="Use Hyperliquid testnet instead of mainnet",
    )

    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_args()

    # Validate assets
    valid_assets = []
    for asset in args.assets:
        asset_upper = asset.upper()
        valid_assets.append(asset_upper)

    if not valid_assets:
        print("Error: No valid assets specified")
        sys.exit(1)

    # Create generator
    generator = LiquidationHeatmapGenerator(
        assets=valid_assets,
        use_testnet=args.testnet,
        output_dir=args.output,
    )

    # Run
    if args.continuous:
        generator.run_continuous(
            interval_seconds=args.interval,
            interactive=args.interactive,
        )
    else:
        output_files = generator.run(
            demo=args.demo,
            interactive=args.interactive,
            max_whale_addresses=args.max_whales,
        )

        if output_files:
            print("\n" + "=" * 60)
            print("OUTPUT FILES")
            print("=" * 60)
            for f in output_files:
                print(f"  {f}")
            print()
        else:
            print("\nNo output files generated")


if __name__ == "__main__":
    main()
