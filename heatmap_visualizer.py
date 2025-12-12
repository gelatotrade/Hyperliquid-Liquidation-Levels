"""
Heatmap Visualization for Liquidation Levels

Creates visual representations of liquidation level data including:
- Price-based heatmaps
- Cumulative liquidation charts
- Interactive Plotly dashboards
"""

import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
import seaborn as sns

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from config import DEFAULT_HEATMAP_CONFIG, TRACKED_ASSETS
from liquidation_calculator import LiquidationHeatmapData, LiquidationCluster


class HeatmapVisualizer:
    """
    Visualizer for liquidation heatmaps

    Supports both static (matplotlib) and interactive (plotly) visualizations
    """

    def __init__(self, config: Optional[Any] = None):
        self.config = config or DEFAULT_HEATMAP_CONFIG
        self._ensure_output_dir()

    def _ensure_output_dir(self):
        """Create output directory if it doesn't exist"""
        if not os.path.exists(self.config.output_dir):
            os.makedirs(self.config.output_dir)

    def create_heatmap(
        self,
        data: LiquidationHeatmapData,
        show_cumulative: bool = True,
        save: bool = True,
        show: bool = False,
    ) -> Optional[str]:
        """
        Create a liquidation heatmap visualization

        Args:
            data: LiquidationHeatmapData object
            show_cumulative: Whether to show cumulative liquidation lines
            save: Whether to save the figure
            show: Whether to display the figure

        Returns:
            Path to saved file if save=True
        """
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(self.config.figure_width, self.config.figure_height),
            gridspec_kw={"width_ratios": [1, 1]},
        )

        # Left panel: Long liquidations (below current price)
        ax_long = axes[0]
        self._plot_liquidation_bars(
            ax_long,
            data.price_levels,
            data.long_values,
            data.current_price,
            is_long=True,
            title=f"{data.coin} Long Liquidations",
        )

        # Right panel: Short liquidations (above current price)
        ax_short = axes[1]
        self._plot_liquidation_bars(
            ax_short,
            data.price_levels,
            data.short_values,
            data.current_price,
            is_long=False,
            title=f"{data.coin} Short Liquidations",
        )

        # Add overall title
        fig.suptitle(
            f"{data.coin} Liquidation Heatmap | Price: ${data.current_price:,.2f} | "
            f"{data.total_positions_analyzed} Positions Analyzed",
            fontsize=14,
            fontweight="bold",
        )

        plt.tight_layout()

        # Save if requested
        filepath = None
        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{data.coin}_liquidation_heatmap_{timestamp}.{self.config.output_format}"
            filepath = os.path.join(self.config.output_dir, filename)
            plt.savefig(filepath, dpi=self.config.dpi, bbox_inches="tight")
            print(f"Heatmap saved to: {filepath}")

        if show:
            plt.show()
        else:
            plt.close()

        return filepath

    def _plot_liquidation_bars(
        self,
        ax: plt.Axes,
        price_levels: np.ndarray,
        values: np.ndarray,
        current_price: float,
        is_long: bool,
        title: str,
    ):
        """Plot horizontal bars for liquidation levels"""
        # Normalize values for color intensity
        max_value = max(np.max(values), 1)
        normalized = values / max_value

        # Create color based on intensity
        if is_long:
            cmap = plt.cm.Greens
            bar_color = "green"
        else:
            cmap = plt.cm.Reds
            bar_color = "red"

        # Plot horizontal bars
        bar_heights = np.diff(price_levels, prepend=price_levels[0] - (price_levels[1] - price_levels[0]))

        for i, (price, value, norm_val) in enumerate(
            zip(price_levels, values, normalized)
        ):
            if value > 0:
                color = cmap(0.3 + 0.7 * norm_val)  # Scale to avoid too light colors
                ax.barh(
                    price,
                    value / 1e6,  # Convert to millions
                    height=bar_heights[i] * 0.8,
                    color=color,
                    alpha=0.8,
                    edgecolor="none",
                )

        # Add current price line
        if self.config.show_current_price_line:
            ax.axhline(
                current_price,
                color=self.config.current_price_color,
                linestyle="--",
                linewidth=2,
                label=f"Current: ${current_price:,.2f}",
            )

        # Styling
        ax.set_xlabel("Liquidation Value ($M)", fontsize=11)
        ax.set_ylabel("Price ($)", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")

        # Format y-axis with dollar signs
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))

        # Grid
        if self.config.show_grid:
            ax.grid(True, alpha=self.config.grid_alpha, linestyle="--")

        ax.legend(loc="best")

    def create_combined_heatmap(
        self,
        data: LiquidationHeatmapData,
        save: bool = True,
        show: bool = False,
    ) -> Optional[str]:
        """
        Create a combined heatmap showing both longs and shorts

        Uses a diverging color scheme (green for longs, red for shorts)
        """
        fig, ax = plt.subplots(
            figsize=(self.config.figure_width, self.config.figure_height)
        )

        price_levels = data.price_levels
        current_price = data.current_price

        # Calculate bar heights
        bar_height = (price_levels[-1] - price_levels[0]) / len(price_levels) * 0.8

        # Plot both sides
        max_value = max(np.max(data.long_values), np.max(data.short_values), 1)

        for i, price in enumerate(price_levels):
            # Long liquidations (left side, green)
            if data.long_values[i] > 0:
                width = data.long_values[i] / 1e6
                ax.barh(
                    price,
                    -width,  # Negative for left side
                    height=bar_height,
                    color="green",
                    alpha=0.3 + 0.7 * (data.long_values[i] / max_value),
                    edgecolor="none",
                )

            # Short liquidations (right side, red)
            if data.short_values[i] > 0:
                width = data.short_values[i] / 1e6
                ax.barh(
                    price,
                    width,
                    height=bar_height,
                    color="red",
                    alpha=0.3 + 0.7 * (data.short_values[i] / max_value),
                    edgecolor="none",
                )

        # Current price line
        ax.axhline(
            current_price,
            color=self.config.current_price_color,
            linestyle="--",
            linewidth=2,
            label=f"Current: ${current_price:,.2f}",
        )

        # Center line
        ax.axvline(0, color="gray", linestyle="-", linewidth=0.5)

        # Labels and title
        ax.set_xlabel("← Long Liquidations ($M) | Short Liquidations ($M) →", fontsize=11)
        ax.set_ylabel("Price ($)", fontsize=11)
        ax.set_title(
            f"{data.coin} Liquidation Levels\n"
            f"Current Price: ${current_price:,.2f} | "
            f"Longs: {int(np.sum(data.long_counts))} | "
            f"Shorts: {int(np.sum(data.short_counts))}",
            fontsize=13,
            fontweight="bold",
        )

        # Format axes
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))

        # Grid
        if self.config.show_grid:
            ax.grid(True, alpha=self.config.grid_alpha, linestyle="--")

        ax.legend(loc="upper right")

        plt.tight_layout()

        # Save if requested
        filepath = None
        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{data.coin}_combined_heatmap_{timestamp}.{self.config.output_format}"
            filepath = os.path.join(self.config.output_dir, filename)
            plt.savefig(filepath, dpi=self.config.dpi, bbox_inches="tight")
            print(f"Combined heatmap saved to: {filepath}")

        if show:
            plt.show()
        else:
            plt.close()

        return filepath

    def create_multi_asset_heatmap(
        self,
        data_list: List[LiquidationHeatmapData],
        save: bool = True,
        show: bool = False,
    ) -> Optional[str]:
        """
        Create a multi-panel heatmap for multiple assets

        Args:
            data_list: List of LiquidationHeatmapData for each asset
            save: Whether to save the figure
            show: Whether to display the figure

        Returns:
            Path to saved file
        """
        n_assets = len(data_list)
        fig, axes = plt.subplots(
            1,
            n_assets,
            figsize=(self.config.figure_width, self.config.figure_height),
            squeeze=False,
        )

        for idx, data in enumerate(data_list):
            ax = axes[0, idx]
            self._plot_mini_heatmap(ax, data)

        fig.suptitle(
            "Hyperliquid Liquidation Heatmap Overview",
            fontsize=14,
            fontweight="bold",
        )

        plt.tight_layout()

        filepath = None
        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"multi_asset_heatmap_{timestamp}.{self.config.output_format}"
            filepath = os.path.join(self.config.output_dir, filename)
            plt.savefig(filepath, dpi=self.config.dpi, bbox_inches="tight")
            print(f"Multi-asset heatmap saved to: {filepath}")

        if show:
            plt.show()
        else:
            plt.close()

        return filepath

    def _plot_mini_heatmap(self, ax: plt.Axes, data: LiquidationHeatmapData):
        """Plot a compact heatmap for multi-asset view"""
        price_levels = data.price_levels
        current_price = data.current_price
        bar_height = (price_levels[-1] - price_levels[0]) / len(price_levels) * 0.8

        max_value = max(np.max(data.long_values), np.max(data.short_values), 1)

        for i, price in enumerate(price_levels):
            if data.long_values[i] > 0:
                width = data.long_values[i] / 1e6
                ax.barh(
                    price,
                    -width,
                    height=bar_height,
                    color="green",
                    alpha=0.4 + 0.6 * (data.long_values[i] / max_value),
                )
            if data.short_values[i] > 0:
                width = data.short_values[i] / 1e6
                ax.barh(
                    price,
                    width,
                    height=bar_height,
                    color="red",
                    alpha=0.4 + 0.6 * (data.short_values[i] / max_value),
                )

        ax.axhline(
            current_price,
            color=self.config.current_price_color,
            linestyle="--",
            linewidth=1.5,
        )
        ax.axvline(0, color="gray", linestyle="-", linewidth=0.5)

        ax.set_title(f"{data.coin}\n${current_price:,.0f}", fontsize=10, fontweight="bold")
        ax.set_xlabel("$M", fontsize=8)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))
        ax.tick_params(axis="both", labelsize=8)

    def create_interactive_heatmap(
        self,
        data: LiquidationHeatmapData,
        save: bool = True,
    ) -> Optional[str]:
        """
        Create an interactive Plotly heatmap

        Args:
            data: LiquidationHeatmapData object
            save: Whether to save as HTML

        Returns:
            Path to saved HTML file
        """
        if not PLOTLY_AVAILABLE:
            print("Plotly not available. Install with: pip install plotly")
            return None

        # Create subplots
        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=("Long Liquidations", "Short Liquidations"),
            horizontal_spacing=0.1,
        )

        # Long liquidations (green)
        fig.add_trace(
            go.Bar(
                y=data.price_levels,
                x=data.long_values / 1e6,
                orientation="h",
                name="Longs",
                marker=dict(
                    color=data.long_values,
                    colorscale="Greens",
                    showscale=False,
                ),
                hovertemplate="Price: $%{y:,.2f}<br>Value: $%{x:.2f}M<extra></extra>",
            ),
            row=1,
            col=1,
        )

        # Short liquidations (red)
        fig.add_trace(
            go.Bar(
                y=data.price_levels,
                x=data.short_values / 1e6,
                orientation="h",
                name="Shorts",
                marker=dict(
                    color=data.short_values,
                    colorscale="Reds",
                    showscale=False,
                ),
                hovertemplate="Price: $%{y:,.2f}<br>Value: $%{x:.2f}M<extra></extra>",
            ),
            row=1,
            col=2,
        )

        # Add current price lines
        for col in [1, 2]:
            fig.add_hline(
                y=data.current_price,
                line_dash="dash",
                line_color="gold",
                annotation_text=f"Current: ${data.current_price:,.2f}",
                annotation_position="top right" if col == 2 else "top left",
                row=1,
                col=col,
            )

        # Update layout
        fig.update_layout(
            title=dict(
                text=f"<b>{data.coin} Liquidation Heatmap</b><br>"
                f"<sub>Price: ${data.current_price:,.2f} | "
                f"Positions: {data.total_positions_analyzed}</sub>",
                x=0.5,
            ),
            showlegend=True,
            height=700,
            template="plotly_dark",
        )

        fig.update_xaxes(title_text="Value ($M)", row=1, col=1)
        fig.update_xaxes(title_text="Value ($M)", row=1, col=2)
        fig.update_yaxes(title_text="Price ($)", tickformat="$,.0f", row=1, col=1)
        fig.update_yaxes(title_text="Price ($)", tickformat="$,.0f", row=1, col=2)

        filepath = None
        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{data.coin}_interactive_heatmap_{timestamp}.html"
            filepath = os.path.join(self.config.output_dir, filename)
            fig.write_html(filepath)
            print(f"Interactive heatmap saved to: {filepath}")

        return filepath

    def create_dashboard(
        self,
        data_list: List[LiquidationHeatmapData],
        save: bool = True,
    ) -> Optional[str]:
        """
        Create a full dashboard with multiple assets

        Args:
            data_list: List of heatmap data for each asset
            save: Whether to save as HTML

        Returns:
            Path to saved HTML file
        """
        if not PLOTLY_AVAILABLE:
            print("Plotly not available. Install with: pip install plotly")
            return None

        n_assets = len(data_list)

        # Create subplots grid
        fig = make_subplots(
            rows=n_assets,
            cols=2,
            subplot_titles=[
                f"{d.coin} Longs" if i % 2 == 0 else f"{d.coin} Shorts"
                for d in data_list
                for i in range(2)
            ],
            vertical_spacing=0.1,
            horizontal_spacing=0.08,
        )

        for row_idx, data in enumerate(data_list, 1):
            # Long liquidations
            fig.add_trace(
                go.Bar(
                    y=data.price_levels,
                    x=data.long_values / 1e6,
                    orientation="h",
                    name=f"{data.coin} Longs",
                    marker=dict(color="green"),
                    showlegend=row_idx == 1,
                ),
                row=row_idx,
                col=1,
            )

            # Short liquidations
            fig.add_trace(
                go.Bar(
                    y=data.price_levels,
                    x=data.short_values / 1e6,
                    orientation="h",
                    name=f"{data.coin} Shorts",
                    marker=dict(color="red"),
                    showlegend=row_idx == 1,
                ),
                row=row_idx,
                col=2,
            )

            # Current price lines
            for col in [1, 2]:
                fig.add_hline(
                    y=data.current_price,
                    line_dash="dash",
                    line_color="gold",
                    row=row_idx,
                    col=col,
                )

        fig.update_layout(
            title=dict(
                text="<b>Hyperliquid Liquidation Dashboard</b>",
                x=0.5,
            ),
            height=400 * n_assets,
            template="plotly_dark",
            showlegend=True,
        )

        filepath = None
        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"liquidation_dashboard_{timestamp}.html"
            filepath = os.path.join(self.config.output_dir, filename)
            fig.write_html(filepath)
            print(f"Dashboard saved to: {filepath}")

        return filepath


class TerminalVisualizer:
    """
    Simple terminal-based visualization for quick viewing
    """

    def __init__(self):
        try:
            from rich.console import Console
            from rich.table import Table
            from rich.panel import Panel

            self.console = Console()
            self.rich_available = True
        except ImportError:
            self.rich_available = False

    def print_summary(self, data: LiquidationHeatmapData, stats: Dict[str, Any]):
        """Print a summary of liquidation data to terminal"""
        if self.rich_available:
            self._print_rich_summary(data, stats)
        else:
            self._print_simple_summary(data, stats)

    def _print_rich_summary(self, data: LiquidationHeatmapData, stats: Dict[str, Any]):
        """Print summary using rich library"""
        from rich.table import Table
        from rich.panel import Panel

        # Create summary panel
        summary = f"""
[bold cyan]{data.coin}[/bold cyan] Liquidation Summary
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[yellow]Current Price:[/yellow] ${data.current_price:,.2f}
[yellow]Positions Analyzed:[/yellow] {data.total_positions_analyzed}

[green]Long Positions:[/green] {stats['long_positions']}
[green]Long Value at Risk:[/green] ${stats['total_long_value_usd']:,.0f}

[red]Short Positions:[/red] {stats['short_positions']}
[red]Short Value at Risk:[/red] ${stats['total_short_value_usd']:,.0f}

[yellow]L/S Ratio:[/yellow] {stats['long_short_ratio']:.2f}
"""
        self.console.print(Panel(summary, title="Liquidation Analysis"))

        # Major levels table
        table = Table(title="Major Liquidation Levels")
        table.add_column("Type", style="cyan")
        table.add_column("Price", style="yellow")
        table.add_column("Value at Risk", style="magenta")

        for price, value in stats.get("major_long_levels", [])[:3]:
            table.add_row("LONG", f"${price:,.2f}", f"${value:,.0f}")

        for price, value in stats.get("major_short_levels", [])[:3]:
            table.add_row("SHORT", f"${price:,.2f}", f"${value:,.0f}")

        self.console.print(table)

    def _print_simple_summary(self, data: LiquidationHeatmapData, stats: Dict[str, Any]):
        """Print summary without rich library"""
        print(f"\n{'='*50}")
        print(f"{data.coin} Liquidation Summary")
        print(f"{'='*50}")
        print(f"Current Price: ${data.current_price:,.2f}")
        print(f"Positions Analyzed: {data.total_positions_analyzed}")
        print(f"\nLong Positions: {stats['long_positions']}")
        print(f"Long Value at Risk: ${stats['total_long_value_usd']:,.0f}")
        print(f"\nShort Positions: {stats['short_positions']}")
        print(f"Short Value at Risk: ${stats['total_short_value_usd']:,.0f}")
        print(f"\nL/S Ratio: {stats['long_short_ratio']:.2f}")

        print(f"\n{'='*50}")
        print("Major Liquidation Levels")
        print(f"{'='*50}")

        print("\nLONG Liquidations (below current price):")
        for price, value in stats.get("major_long_levels", [])[:3]:
            print(f"  ${price:,.2f}: ${value:,.0f} at risk")

        print("\nSHORT Liquidations (above current price):")
        for price, value in stats.get("major_short_levels", [])[:3]:
            print(f"  ${price:,.2f}: ${value:,.0f} at risk")

        print()


def get_visualizer() -> HeatmapVisualizer:
    """Factory function to create a visualizer instance"""
    return HeatmapVisualizer(config=DEFAULT_HEATMAP_CONFIG)


def get_terminal_visualizer() -> TerminalVisualizer:
    """Factory function to create a terminal visualizer"""
    return TerminalVisualizer()
