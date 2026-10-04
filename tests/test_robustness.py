import numpy as np
import pandas as pd
import pytest

from optimization.robustness import (
    NeighborhoodConfig,
    add_neighborhood_robustness,
)


def make_plateau_results() -> pd.DataFrame:
    """
    Create a one-dimensional RSI neighborhood around RSI 12.

    All other parameters remain constant.
    """

    return pd.DataFrame(
        {
            "rsi_length": [
                8,
                10,
                12,
                14,
                16,
            ],
            "oversold": [
                30.0,
                30.0,
                30.0,
                30.0,
                30.0,
            ],
            "overbought": [
                70.0,
                70.0,
                70.0,
                70.0,
                70.0,
            ],
            "max_bars_between_dips": [
                7,
                7,
                7,
                7,
                7,
            ],
            "sharpe": [
                1.20,
                1.40,
                1.50,
                1.35,
                1.15,
            ],
        }
    )


def test_direct_neighbors_use_adjacent_grid_values() -> None:
    results = make_plateau_results()

    robust = add_neighborhood_robustness(
        results,
        NeighborhoodConfig(
            metric="sharpe",
            min_neighbors=2,
        ),
    )

    center = robust[
        robust["rsi_length"] == 12
    ].iloc[0]

    # RSI 12 has immediate neighbors:
    #
    # RSI 10
    # RSI 14
    #
    assert center["neighbor_count"] == 2


def test_plateau_candidate_has_high_robustness() -> None:
    results = make_plateau_results()

    robust = add_neighborhood_robustness(
        results,
        NeighborhoodConfig(
            metric="sharpe",
            min_neighbors=2,
            relative_performance_floor=0.75,
        ),
    )

    center = robust[
        robust["rsi_length"] == 12
    ].iloc[0]

    expected_median = np.median(
        [
            1.40,
            1.35,
        ]
    )

    expected_ratio = (
        expected_median / 1.50
    )

    assert center[
        "neighbor_metric_median"
    ] == pytest.approx(
        expected_median
    )

    assert center[
        "healthy_neighbor_fraction"
    ] == pytest.approx(
        1.0
    )

    assert center[
        "neighborhood_ratio"
    ] == pytest.approx(
        expected_ratio
    )

    assert center[
        "neighborhood_robustness"
    ] == pytest.approx(
        expected_ratio
    )


def test_isolated_spike_has_low_robustness() -> None:
    results = make_plateau_results()

    results.loc[
        results["rsi_length"] == 12,
        "sharpe",
    ] = 5.0

    robust = add_neighborhood_robustness(
        results,
        NeighborhoodConfig(
            metric="sharpe",
            min_neighbors=2,
            relative_performance_floor=0.75,
        ),
    )

    center = robust[
        robust["rsi_length"] == 12
    ].iloc[0]

    assert center[
        "healthy_neighbor_fraction"
    ] == pytest.approx(
        0.0
    )

    assert center[
        "neighborhood_robustness"
    ] == pytest.approx(
        0.0
    )


def test_insufficient_neighbors_produces_nan_robustness() -> None:
    results = make_plateau_results()

    robust = add_neighborhood_robustness(
        results,
        NeighborhoodConfig(
            metric="sharpe",

            # Center only has two direct neighbors.
            min_neighbors=3,
        ),
    )

    center = robust[
        robust["rsi_length"] == 12
    ].iloc[0]

    assert np.isnan(
        center[
            "neighborhood_robustness"
        ]
    )


def test_nan_neighbor_metric_is_ignored() -> None:
    results = make_plateau_results()

    results.loc[
        results["rsi_length"] == 10,
        "sharpe",
    ] = np.nan

    robust = add_neighborhood_robustness(
        results,
        NeighborhoodConfig(
            metric="sharpe",
            min_neighbors=1,
        ),
    )

    center = robust[
        robust["rsi_length"] == 12
    ].iloc[0]

    # RSI 10 is invalid.
    # RSI 14 remains valid.
    assert center[
        "neighbor_count"
    ] == 2

    assert center[
        "valid_neighbor_count"
    ] == 1


def test_original_results_are_not_modified() -> None:
    results = make_plateau_results()

    original = results.copy(
        deep=True
    )

    add_neighborhood_robustness(
        results,
        NeighborhoodConfig(
            metric="sharpe",
            min_neighbors=2,
        ),
    )

    pd.testing.assert_frame_equal(
        results,
        original,
    )


def test_empty_results_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="optimization results cannot be empty",
    ):
        add_neighborhood_robustness(
            pd.DataFrame()
        )


def test_invalid_min_neighbors_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="min_neighbors must be at least 1",
    ):
        NeighborhoodConfig(
            min_neighbors=0
        )


def test_invalid_relative_floor_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="relative_performance_floor",
    ):
        NeighborhoodConfig(
            relative_performance_floor=1.25
        )