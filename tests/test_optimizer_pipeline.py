import numpy as np
import pandas as pd
import pytest

from ta.momentum import RSIIndicator

from strats.double_dip import (
    rsi_double_dip_targets,
)

from backtesting.engine import (
    BacktestConfig,
    run_backtest,
)

from backtesting.metrics import (
    calculate_metrics,
)

from optimization.search_space import (
    DoubleDipSearchSpace,
)

from optimization.static import (
    optimize_double_dip,
)

from optimization.robustness import (
    NeighborhoodConfig,
    add_neighborhood_robustness,
)

from optimization.selection import (
    SelectionConfig,
    select_double_dip_candidate,
)


def make_optimizer_market_data(
    periods: int = 300,
) -> pd.DataFrame:
    """
    Build deterministic synthetic market data with repeated
    downward/upward movements.

    This is not intended to simulate a real stock perfectly.

    Its purpose is to provide enough changing momentum for
    multiple RSI configurations and completed Double Dip trades
    to pass through the complete optimization pipeline.
    """

    index = pd.date_range(
        "2020-01-01",
        periods=periods,
        freq="D",
    )

    x = np.arange(periods)

    close = (
        100.0
        + (x * 0.05)
        + 10.0 * np.sin(
            2 * np.pi * x / 30
        )
        + 3.0 * np.sin(
            2 * np.pi * x / 9
        )
    )

    open_ = close * 0.999

    return pd.DataFrame(
        {
            "Open": open_,
            "Close": close,
        },
        index=index,
    )


def test_complete_static_optimizer_pipeline_with_robustness() -> None:

    # ========================================================
    # 1. MARKET DATA
    # ========================================================

    data = make_optimizer_market_data()

    # ========================================================
    # 2. SEARCH SPACE
    # ========================================================
    #
    # Keep this significantly smaller than the production
    # search grid so the integration test remains fast.
    #
    # Using at least three values in some dimensions is useful
    # for neighborhood robustness because it creates interior
    # candidates with neighbors on both sides.
    # ========================================================

    search_space = DoubleDipSearchSpace(
        rsi_lengths=(
            5,
            8,
            14,
        ),
        oversold_levels=(
            25.0,
            30.0,
            35.0,
        ),
        overbought_levels=(
            65.0,
            70.0,
            75.0,
        ),
        max_bars_between_dips=(
            5,
            10,
            15,
        ),
    )

    expected_candidate_count = (
        3
        * 3
        * 3
        * 3
    )

    assert search_space.candidate_count() == (
        expected_candidate_count
    )

    # ========================================================
    # 3. BACKTEST CONFIGURATION
    # ========================================================

    backtest_config = BacktestConfig(
        initial_cash=100_000.0,

        # Zero costs simplify this integration test.
        commission_rate=0.0,
        slippage_rate=0.0,

        close_open_position_at_end=True,
    )

    # ========================================================
    # 4. STATIC OPTIMIZATION
    # ========================================================

    raw_results = optimize_double_dip(
        data=data,
        search_space=search_space,
        backtest_config=backtest_config,
        periods_per_year=252,
    )

    assert len(raw_results) == (
        expected_candidate_count
    )

    assert "sharpe" in raw_results.columns
    assert "closed_trades" in raw_results.columns
    assert "total_return" in raw_results.columns
    assert "final_equity" in raw_results.columns

    # Robustness should NOT exist yet.
    assert (
        "neighborhood_robustness"
        not in raw_results.columns
    )

    # ========================================================
    # 5. NEIGHBORHOOD ROBUSTNESS
    # ========================================================

    neighborhood_config = NeighborhoodConfig(
        metric="sharpe",

        # Keep this relatively permissive for an integration
        # test. Dedicated robustness unit tests test stricter
        # behavior.
        min_neighbors=2,

        relative_performance_floor=0.75,

        # Only modify one parameter at a time when determining
        # direct neighbors.
        include_diagonal_neighbors=False,

        require_positive_neighbor_metric=True,
    )

    robust_results = add_neighborhood_robustness(
        results=raw_results,
        config=neighborhood_config,
    )

    # ========================================================
    # 6. VERIFY ROBUSTNESS COLUMNS
    # ========================================================

    expected_robustness_columns = {
        "neighbor_count",
        "valid_neighbor_count",
        "neighbor_metric_mean",
        "neighbor_metric_median",
        "neighbor_metric_min",
        "neighbor_metric_std",
        "healthy_neighbor_count",
        "healthy_neighbor_fraction",
        "neighborhood_ratio",
        "neighborhood_robustness",
    }

    assert expected_robustness_columns.issubset(
        robust_results.columns
    )

    # Robustness enrichment should not create or remove
    # candidates.
    assert len(robust_results) == (
        len(raw_results)
    )

    # And it must not mutate the original static results.
    assert (
        "neighborhood_robustness"
        not in raw_results.columns
    )

    # ========================================================
    # 7. VERIFY AT LEAST ONE ROBUSTNESS SCORE EXISTS
    # ========================================================
    #
    # Some candidates may legitimately receive NaN if they do
    # not have enough valid neighbors.
    #
    # But the entire grid should not be unusable.
    # ========================================================

    valid_robustness = robust_results[
        "neighborhood_robustness"
    ].dropna()

    assert not valid_robustness.empty

    assert (
        valid_robustness >= 0
    ).all()

    assert (
        valid_robustness <= 1
    ).all()

     # ========================================================
    # 8. SELECTION
    # ========================================================

    selection_config = SelectionConfig(
        # Keep low for synthetic integration data.
        min_closed_trades=1,

        ranking_metric="sharpe",

        # Some synthetic candidates could lose money.
        require_positive_return=False,

        # Require at least some neighborhood support.
        #
        # Keep permissive here because robustness behavior
        # itself is covered by its dedicated unit tests.
        min_neighborhood_robustness=0.10,
    )

    selected = select_double_dip_candidate(
        results=robust_results,
        config=selection_config,
    )

    parameters = selected.parameters

    # ========================================================
    # 9. VERIFY SELECTED PARAMETERS CAME FROM SEARCH SPACE
    # ========================================================

    assert parameters.rsi_length in search_space.rsi_lengths

    assert parameters.oversold in search_space.oversold_levels

    assert parameters.overbought in search_space.overbought_levels

    assert (
        parameters.max_bars_between_dips
        in search_space.max_bars_between_dips
    )

    # ========================================================
    # 10. FIND SELECTED CANDIDATE IN ROBUST RESULT TABLE
    # ========================================================

    matching_rows = robust_results[
        (
            robust_results["rsi_length"]
            == parameters.rsi_length
        )
        &
        (
            robust_results["oversold"]
            == parameters.oversold
        )
        &
        (
            robust_results["overbought"]
            == parameters.overbought
        )
        &
        (
            robust_results["max_bars_between_dips"]
            == parameters.max_bars_between_dips
        )
    ]

    # Every parameter configuration should occur exactly once.
    assert len(matching_rows) == 1

    selected_row = matching_rows.iloc[0]

    # ========================================================
    # 11. VERIFY ROBUSTNESS REQUIREMENT
    # ========================================================

    assert not np.isnan(
        selected_row["neighborhood_robustness"]
    )

    assert (
        selected_row["neighborhood_robustness"]
        >= selection_config.min_neighborhood_robustness
    )

    # There should also be enough valid neighbors for the
    # robustness calculation to have actually been produced.
    assert (
        selected_row["valid_neighbor_count"]
        >= neighborhood_config.min_neighbors
    )

    # ========================================================
    # 12. VERIFY SELECTOR METRICS MATCH SELECTED ROW
    # ========================================================

    assert selected.ranking_metric == "sharpe"

    assert selected.ranking_value == pytest.approx(
        selected_row["sharpe"]
    )

    assert selected.closed_trades == int(
        selected_row["closed_trades"]
    )

    assert selected.total_return == pytest.approx(
        selected_row["total_return"]
    )

    assert selected.annualized_return == pytest.approx(
        selected_row["annualized_return"]
    )

    assert selected.max_drawdown == pytest.approx(
        selected_row["max_drawdown"]
    )

    assert selected.sharpe == pytest.approx(
        selected_row["sharpe"]
    )

    assert selected.profit_factor == pytest.approx(
        selected_row["profit_factor"]
    )

    # ========================================================
    # 13. RE-CALCULATE RSI WITH SELECTED PARAMETERS
    # ========================================================

    rsi = RSIIndicator(
        close=data["Close"],
        window=parameters.rsi_length,
    ).rsi()

    # ========================================================
    # 14. RE-GENERATE DOUBLE DIP TARGETS
    # ========================================================

    targets = rsi_double_dip_targets(
        rsi=rsi,
        oversold=parameters.oversold,
        overbought=parameters.overbought,
        max_bars_between_dips=(
            parameters.max_bars_between_dips
        ),
    )

    # Strategy output must remain aligned with market data.
    assert targets.index.equals(data.index)

    assert targets.isin([0, 1]).all()

    # ========================================================
    # 15. RE-RUN SELECTED CONFIGURATION THROUGH ENGINE
    # ========================================================

    result = run_backtest(
        data=data,
        target_at_close=targets,
        config=backtest_config,
    )

    # ========================================================
    # 16. RE-CALCULATE METRICS
    # ========================================================

    metrics = calculate_metrics(
        equity=result.equity,
        trades=result.trades,
        periods_per_year=252,
    )

    # ========================================================
    # 17. VERIFY OPTIMIZER REPRODUCIBILITY
    # ========================================================
    #
    # This is one of the most important integration checks.
    #
    # static optimizer:
    #
    #     selected parameters
    #         ↓
    #     recorded performance
    #
    # must be identical to:
    #
    #     selected parameters
    #         ↓
    #     strategy
    #         ↓
    #     engine
    #         ↓
    #     metrics
    #
    # If these disagree, the optimizer and normal execution
    # pipeline are not equivalent.
    # ========================================================

    assert metrics["total_return"] == pytest.approx(
        selected_row["total_return"]
    )

    assert metrics["annualized_return"] == pytest.approx(
        selected_row["annualized_return"]
    )

    assert metrics["max_drawdown"] == pytest.approx(
        selected_row["max_drawdown"]
    )

    assert metrics["closed_trades"] == int(
        selected_row["closed_trades"]
    )

    assert metrics["winning_trades"] == int(
        selected_row["winning_trades"]
    )

    assert metrics["losing_trades"] == int(
        selected_row["losing_trades"]
    )

    assert metrics["net_profit"] == pytest.approx(
        selected_row["net_profit"]
    )

    assert result.equity.iloc[-1] == pytest.approx(
        selected_row["final_equity"]
    )

    # ========================================================
    # 18. HANDLE METRICS THAT MAY BE NaN OR INFINITY
    # ========================================================

    if np.isnan(selected_row["sharpe"]):
        assert np.isnan(metrics["sharpe"])
    else:
        assert metrics["sharpe"] == pytest.approx(
            selected_row["sharpe"]
        )

    if np.isnan(selected_row["profit_factor"]):
        assert np.isnan(metrics["profit_factor"])

    elif np.isinf(selected_row["profit_factor"]):
        assert np.isinf(metrics["profit_factor"])

    else:
        assert metrics["profit_factor"] == pytest.approx(
            selected_row["profit_factor"]
        )

    # ========================================================
    # 19. ROBUSTNESS MUST NOT CHANGE STRATEGY PERFORMANCE
    # ========================================================
    #
    # Neighborhood analysis annotates candidates.
    #
    # It must NOT modify the underlying backtest results.
    # ========================================================

    raw_matching_rows = raw_results[
        (
            raw_results["rsi_length"]
            == parameters.rsi_length
        )
        &
        (
            raw_results["oversold"]
            == parameters.oversold
        )
        &
        (
            raw_results["overbought"]
            == parameters.overbought
        )
        &
        (
            raw_results["max_bars_between_dips"]
            == parameters.max_bars_between_dips
        )
    ]

    assert len(raw_matching_rows) == 1

    raw_row = raw_matching_rows.iloc[0]

    assert selected_row["total_return"] == pytest.approx(
        raw_row["total_return"]
    )

    assert selected_row["final_equity"] == pytest.approx(
        raw_row["final_equity"]
    )

    assert selected_row["closed_trades"] == (
        raw_row["closed_trades"]
    )

    assert selected_row["max_drawdown"] == pytest.approx(
        raw_row["max_drawdown"]
    )

    # ========================================================
    # 20. FINAL PIPELINE SANITY CHECKS
    # ========================================================

    assert len(result.equity) == len(data)

    assert len(result.positions) == len(data)

    assert result.equity.index.equals(
        data.index
    )

    assert result.positions.index.equals(
        data.index
    )

    assert (result.equity > 0).all()