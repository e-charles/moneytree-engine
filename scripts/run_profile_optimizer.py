from __future__ import annotations

import sys
from pathlib import Path

# Adds the parent directory (/workspaces/moneytree-optimizer) to the path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import cProfile
import pstats
import time

import yfinance as yf

from optimization.search_space import (
    default_double_dip_search_space,
)
from optimization.robustness import (
    NeighborhoodConfig,
)
from optimization.selection import (
    SelectionConfig,
)
from optimization.walk_forward import (
    WalkForwardConfig,
    walk_forward_optimize_double_dip,
)


def load_data():
    """
    Download historical AAPL daily data.
    """

    data = yf.download(
        "AAPL",
        start="2023-01-01",
        end="2026-01-01",
        interval="1d",
        auto_adjust=True,
        progress=False,
        multi_level_index=False,
    )

    if data.empty:
        raise ValueError(
            "No historical data was downloaded"
        )

    return data


def run_optimizer(data):
    """
    Run the same walk-forward optimization that we want
    to profile.
    """

    # ========================================================
    # 1. SEARCH SPACE
    # ========================================================

    search_space = (
        default_double_dip_search_space()
    )

    # ========================================================
    # 2. WALK-FORWARD CONFIG
    # ========================================================

    walk_forward_config = WalkForwardConfig(
        train_bars=504,
        test_bars=63,
        expanding=False,
    )

    # ========================================================
    # 3. NEIGHBORHOOD ROBUSTNESS
    # ========================================================

    neighborhood_config = NeighborhoodConfig(
        metric="sharpe",
        min_neighbors=3,
        relative_performance_floor=0.75,
        include_diagonal_neighbors=False,
    )

    # ========================================================
    # 4. SELECTION
    # ========================================================

    selection_config = SelectionConfig(
        min_closed_trades=1,
        ranking_metric="sharpe",
        max_allowed_drawdown=0.10,
        require_positive_return=True,
        min_neighborhood_robustness=0.60,
    )

    # ========================================================
    # 5. WALK-FORWARD OPTIMIZATION
    # ========================================================

    result = walk_forward_optimize_double_dip(
        data=data,
        search_space=search_space,
        walk_forward_config=walk_forward_config,
        selection_config=selection_config,
        neighborhood_config=neighborhood_config,
    )

    return result


def main():
    # ========================================================
    # LOAD DATA
    # ========================================================

    print("Loading historical data...")

    data = load_data()

    print(
        f"Loaded {len(data):,} bars"
    )

    print(
        f"Start: {data.index[0]}"
    )

    print(
        f"End:   {data.index[-1]}"
    )

    print()

    # ========================================================
    # CREATE PROFILER
    # ========================================================

    profiler = cProfile.Profile()

    # Also measure ordinary wall-clock runtime so we have
    # an easy benchmark to compare against previous runs.
    start = time.perf_counter()

    # ========================================================
    # START PROFILING
    # ========================================================

    profiler.enable()

    result = run_optimizer(
        data=data
    )

    profiler.disable()

    # ========================================================
    # END TIMING
    # ========================================================

    elapsed = (
        time.perf_counter()
        - start
    )

    # ========================================================
    # PRINT WALK-FORWARD RESULTS
    # ========================================================

    print()
    print("=" * 80)
    print("WALK-FORWARD RESULTS")
    print("=" * 80)

    print(
        result.summary.to_string(
            index=False
        )
    )

    # ========================================================
    # PRINT TOTAL RUNTIME
    # ========================================================

    print()
    print("=" * 80)
    print("RUNTIME")
    print("=" * 80)

    print(
        f"Total runtime: {elapsed:.2f} seconds"
    )

    print()

    # ========================================================
    # PROFILER RESULTS
    # ========================================================
    #
    # strip_dirs():
    #     Removes long source paths from output.
    #
    # sort_stats("cumulative"):
    #     Functions are ranked by the total amount of time
    #     spent inside the function AND everything it calls.
    #
    # print_stats(30):
    #     Show only the 30 most expensive entries.
    # ========================================================

    stats = pstats.Stats(
        profiler
    )

    stats.strip_dirs()

    stats.sort_stats(
        "cumulative"
    )

    print()
    print("=" * 80)
    print("TOP 30 FUNCTIONS BY CUMULATIVE TIME")
    print("=" * 80)

    stats.print_stats(
        30
    )

    # ========================================================
    # SAVE PROFILE
    # ========================================================
    #
    # This lets us inspect the same run later without having
    # to re-run the optimizer.
    # ========================================================

    profiler.dump_stats(
        "optimizer_profile.prof"
    )

    print()
    print(
        "Profile saved to: optimizer_profile.prof"
    )


if __name__ == "__main__":
    main()