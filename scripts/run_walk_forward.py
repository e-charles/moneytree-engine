import sys
from pathlib import Path
import time
# Adds the parent directory (/workspaces/moneytree-optimizer) to the path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import yfinance as yf

from optimization.search_space import (
    default_double_dip_search_space,
)
from optimization.selection import (
    SelectionConfig,
)
from optimization.robustness import (
    NeighborhoodConfig,
)
from optimization.walk_forward import (
    WalkForwardConfig,
    walk_forward_optimize_double_dip,
)


# ============================================================
# 1. DOWNLOAD HISTORICAL DATA
# ============================================================

data = yf.download(
    "AAPL",
    start="2023-01-01",
    end="2026-01-01",
    interval="1d",
    auto_adjust=True,
    progress=False,
    multi_level_index=False,
)

print(data.head())
print(data.tail())
print(f"Bars: {len(data)}")


# ============================================================
# 2. SEARCH SPACE
# ============================================================

search_space = default_double_dip_search_space()


# ============================================================
# 3. WALK-FORWARD CONFIGURATION
# ============================================================

walk_forward_config = WalkForwardConfig(
    train_bars=504,
    test_bars=63,
    expanding=False,
)


# ============================================================
# 4. ROBUSTNESS
# ============================================================

neighborhood_config = NeighborhoodConfig(
    metric="sharpe",
    min_neighbors=1,
    relative_performance_floor=0.45,
    include_diagonal_neighbors=True,
)


# ============================================================
# 5. SELECTION
# ============================================================

selection_config = SelectionConfig(
    min_closed_trades=1,
    ranking_metric="sharpe",
    max_allowed_drawdown=0.10,
    require_positive_return=True,
    min_neighborhood_robustness=0.50,
)


# ============================================================
# 6. WALK-FORWARD OPTIMIZATION
# ============================================================
start = time.perf_counter()

result = walk_forward_optimize_double_dip(
    data=data,
    search_space=search_space,
    walk_forward_config=walk_forward_config,
    selection_config=selection_config,
    neighborhood_config=neighborhood_config,
)

elapsed = time.perf_counter() - start

# ============================================================
# 7. RESULTS
# ============================================================

print(result.summary)

print()
print(f"Runtime: {elapsed:.2f} seconds")