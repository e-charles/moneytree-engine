# /optimization/robustness.py

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd


PARAMETER_COLUMNS = (
    "rsi_length",
    "oversold",
    "overbought",
    "max_bars_between_dips",
)


ParameterValue = float | int
ParameterKey = tuple[
    ParameterValue,
    ParameterValue,
    ParameterValue,
    ParameterValue,
]


@dataclass(frozen=True)
class NeighborhoodConfig:
    """
    Configuration controlling parameter-neighborhood robustness.

    metric:
        Performance metric used to evaluate the neighborhood.

    min_neighbors:
        Minimum number of valid neighboring configurations
        required before neighborhood robustness is considered
        valid.

    relative_performance_floor:
        A neighbor is considered healthy when its metric is at
        least this fraction of the center candidate's metric.

        Example:

            center Sharpe = 1.50
            floor = 0.75

            threshold = 1.125

    include_diagonal_neighbors:
        If False, neighbors differ by exactly one grid step
        along exactly one parameter dimension.

        If True, neighbors may differ by one grid step across
        multiple parameter dimensions simultaneously.

    require_positive_neighbor_metric:
        If True, only positive neighboring metric values are
        considered valid.
    """

    metric: str = "sharpe"

    min_neighbors: int = 3

    relative_performance_floor: float = 0.75

    include_diagonal_neighbors: bool = False

    require_positive_neighbor_metric: bool = True

    def __post_init__(self) -> None:

        if self.min_neighbors < 1:
            raise ValueError(
                "min_neighbors must be at least 1"
            )

        if not (
            0
            < self.relative_performance_floor
            <= 1
        ):
            raise ValueError(
                "relative_performance_floor must be "
                "greater than 0 and less than or equal to 1"
            )


def _validate_results(
    results: pd.DataFrame,
    config: NeighborhoodConfig,
) -> None:
    """
    Validate static optimizer results before neighborhood
    analysis.
    """

    if results.empty:
        raise ValueError(
            "optimization results cannot be empty"
        )

    required_columns = {
        *PARAMETER_COLUMNS,
        config.metric,
    }

    missing = required_columns.difference(
        results.columns
    )

    if missing:
        raise ValueError(
            "Missing required optimization columns: "
            f"{sorted(missing)}"
        )

    if results.duplicated(
        subset=list(PARAMETER_COLUMNS)
    ).any():
        raise ValueError(
            "optimization results contain duplicate "
            "parameter configurations"
        )


def _ordered_parameter_values(
    results: pd.DataFrame,
) -> dict[str, list[ParameterValue]]:
    """
    Extract the ordered values represented along each
    parameter dimension.

    Grid adjacency is based on position in these lists rather
    than numerical distance.
    """

    return {
        column: sorted(
            results[
                column
            ].dropna().unique().tolist()
        )
        for column in PARAMETER_COLUMNS
    }


def _parameter_grid_positions(
    values: dict[str, list[ParameterValue]],
) -> dict[str, dict[ParameterValue, int]]:
    """
    Map each parameter value to its position on its grid axis.

    Example:

        rsi_length = [5, 8, 14]

    becomes:

        {
            5: 0,
            8: 1,
            14: 2,
        }

    This allows unevenly spaced parameter grids while still
    treating adjacent choices as one grid step apart.
    """

    return {
        column: {
            value: position
            for position, value
            in enumerate(column_values)
        }
        for column, column_values
        in values.items()
    }


def _build_candidate_lookup(
    results: pd.DataFrame,
) -> dict[ParameterKey, int]:
    """
    Build direct lookup from a complete parameter configuration
    to its row index.

    Example:

        (
            14,
            30.0,
            70.0,
            5,
        )

    maps directly to the optimizer-result row containing that
    configuration.
    """

    lookup: dict[ParameterKey, int] = {}

    for row_index, row in results.iterrows():

        key: ParameterKey = (
            row["rsi_length"],
            row["oversold"],
            row["overbought"],
            row["max_bars_between_dips"],
        )

        lookup[key] = row_index

    return lookup


def _neighbor_axis_values(
    column: str,
    value: ParameterValue,
    values: dict[str, list[ParameterValue]],
    positions: dict[
        str,
        dict[ParameterValue, int],
    ],
) -> list:
    """
    Return the center value plus its immediate previous and next
    values along one grid dimension.

    Example:

        values:
            [5, 8, 14, 21]

        center:
            14

        returns:
            [14, 8, 21]

    At a boundary:

        center:
            5

        returns:
            [5, 8]
    """

    axis_values = values[column]

    center_position = positions[column][
        value
    ]

    choices: list[ParameterValue] = [
        value
    ]

    previous_position = (
        center_position - 1
    )

    next_position = (
        center_position + 1
    )

    if previous_position >= 0:
        choices.append(
            axis_values[
                previous_position
            ]
        )

    if next_position < len(axis_values):
        choices.append(
            axis_values[
                next_position
            ]
        )

    return choices


def _find_neighbor_indices(
    candidate: pd.Series,
    values: dict[str, list[ParameterValue]],
    positions: dict[
        str,
        dict[ParameterValue, int],
    ],
    candidate_lookup: dict[ParameterKey, int],
    include_diagonal_neighbors: bool,
) -> list:
    """
    Find immediate parameter-grid neighbors using direct
    dictionary lookup.

    This avoids comparing the candidate against every row in
    the optimizer result table.

    Direct-neighbor mode:
        Exactly one parameter differs by one grid step.

    Diagonal-neighbor mode:
        One or more parameters may differ by one grid step,
        while no parameter may differ by more than one step.
    """

    center_key: ParameterKey = (
        candidate["rsi_length"],
        candidate["oversold"],
        candidate["overbought"],
        candidate["max_bars_between_dips"],
    )

    # ---------------------------------------------------------
    # FAST PATH: axis-aligned neighbors only
    # ---------------------------------------------------------
    #
    # Four dimensions means at most:
    #
    #     4 × 2 = 8
    #
    # possible direct neighbors.
    # ---------------------------------------------------------

    if not include_diagonal_neighbors:

        neighbor_indices: list[int] = []

        for dimension, column in enumerate(
            PARAMETER_COLUMNS
        ):
            current_value = center_key[
                dimension
            ]

            current_position = positions[
                column
            ][current_value]

            axis_values = values[
                column
            ]

            for offset in (-1, 1):

                neighbor_position = (
                    current_position + offset
                )

                if not (
                    0
                    <= neighbor_position
                    < len(axis_values)
                ):
                    continue

                neighbor_key_list = list(
                    center_key
                )

                neighbor_key_list[
                    dimension
                ] = axis_values[
                    neighbor_position
                ]

                neighbor_key: ParameterKey = tuple(
                    neighbor_key_list
                )

                neighbor_index = (
                    candidate_lookup.get(
                        neighbor_key
                    )
                )

                # A theoretical grid neighbor may not actually
                # exist in the result table.
                #
                # Example:
                #
                # oversold >= overbought combinations are
                # excluded by the Double Dip search space.
                if neighbor_index is not None:
                    neighbor_indices.append(
                        neighbor_index
                    )

        return neighbor_indices

    # ---------------------------------------------------------
    # DIAGONAL MODE
    # ---------------------------------------------------------
    #
    # Build the set of:
    #
    #     previous / center / next
    #
    # values along each dimension, then generate their Cartesian
    # product.
    #
    # With four dimensions there are at most:
    #
    #     3^4 - 1 = 80
    #
    # possible immediate neighbors.
    #
    # That is still far smaller than scanning all 2,080
    # candidates.
    # ---------------------------------------------------------

    axis_choices = [
        _neighbor_axis_values(
            column=column,
            value=candidate[column],
            values=values,
            positions=positions,
        )
        for column in PARAMETER_COLUMNS
    ]

    neighbor_indices = []

    for combination in product(
        *axis_choices
    ):

        neighbor_key: ParameterKey = (
            combination[0],
            combination[1],
            combination[2],
            combination[3],
        )

        # Exclude the center candidate itself.
        if neighbor_key == center_key:
            continue

        neighbor_index = (
            candidate_lookup.get(
                neighbor_key
            )
        )

        if neighbor_index is not None:
            neighbor_indices.append(
                neighbor_index
            )

    return neighbor_indices


def add_neighborhood_robustness(
    results: pd.DataFrame,
    config: NeighborhoodConfig | None = None,
) -> pd.DataFrame:
    """
    Add parameter-neighborhood robustness diagnostics to static
    optimization results.

    The original DataFrame is not modified.

    Added columns
    -------------

    neighbor_count:
        Number of immediate parameter-grid neighbors that exist.

    valid_neighbor_count:
        Number of neighbors with a usable metric.

    neighbor_metric_mean:
        Mean metric across valid neighboring configurations.

    neighbor_metric_median:
        Median metric across valid neighboring configurations.

    neighbor_metric_min:
        Worst valid neighboring metric.

    neighbor_metric_std:
        Population standard deviation across valid neighboring
        metrics.

    healthy_neighbor_count:
        Number of valid neighbors achieving at least:

            center metric
            *
            relative_performance_floor

    healthy_neighbor_fraction:
        Fraction of valid neighbors considered healthy.

    neighborhood_ratio:
        Median neighboring performance divided by center
        performance.

    neighborhood_robustness:
        Combined robustness score:

            healthy_neighbor_fraction
            *
            clipped neighborhood_ratio

        The score lies between 0 and 1.

        NaN means there was insufficient information to assess
        robustness.
    """

    if config is None:
        config = NeighborhoodConfig()

    _validate_results(
        results=results,
        config=config,
    )

    # Reset index so dictionary lookup and iloc-style candidate
    # relationships remain simple and predictable.
    output = (
        results.copy()
        .reset_index(drop=True)
    )

    # ---------------------------------------------------------
    # Precompute grid structures ONCE
    # ---------------------------------------------------------

    values = _ordered_parameter_values(
        output
    )

    positions = _parameter_grid_positions(
        values
    )

    candidate_lookup = (
        _build_candidate_lookup(
            output
        )
    )

    # ---------------------------------------------------------
    # Result containers
    # ---------------------------------------------------------

    neighbor_counts: list[int] = []
    valid_neighbor_counts: list[int] = []

    neighbor_means: list[float] = []
    neighbor_medians: list[float] = []
    neighbor_minimums: list[float] = []
    neighbor_stds: list[float] = []

    healthy_counts: list[int] = []
    healthy_fractions: list[float] = []

    neighborhood_ratios: list[float] = []
    robustness_scores: list[float] = []

    metric = config.metric

    # ---------------------------------------------------------
    # Evaluate neighborhood around every candidate
    # ---------------------------------------------------------

    for row_index in output.index:

        candidate = output.loc[
            row_index
        ]

        center_metric = float(
            candidate[metric]
        )

        neighbor_indices = (
            _find_neighbor_indices(
                candidate=candidate,
                values=values,
                positions=positions,
                candidate_lookup=(
                    candidate_lookup
                ),
                include_diagonal_neighbors=(
                    config.include_diagonal_neighbors
                ),
            )
        )

        neighbor_count = len(
            neighbor_indices
        )

        neighbor_counts.append(
            neighbor_count
        )

        # -----------------------------------------------------
        # Neighbor performance
        # -----------------------------------------------------

        neighbor_metrics = (
            output.loc[
                neighbor_indices,
                metric,
            ]
            .astype(float)
        )

        # Remove NaN and +/- infinity.
        valid_mask = np.isfinite(
            neighbor_metrics.to_numpy()
        )

        valid_neighbor_metrics = (
            neighbor_metrics.iloc[
                np.flatnonzero(
                    valid_mask
                )
            ]
        )

        if (
            config
            .require_positive_neighbor_metric
        ):
            valid_neighbor_metrics = (
                valid_neighbor_metrics[
                    valid_neighbor_metrics
                    > 0
                ]
            )

        valid_neighbor_count = len(
            valid_neighbor_metrics
        )

        valid_neighbor_counts.append(
            valid_neighbor_count
        )

        # -----------------------------------------------------
        # Insufficient information
        # -----------------------------------------------------

        if (
            valid_neighbor_count
            < config.min_neighbors
            or not np.isfinite(
                center_metric
            )
            or center_metric <= 0
        ):
            neighbor_means.append(
                np.nan
            )

            neighbor_medians.append(
                np.nan
            )

            neighbor_minimums.append(
                np.nan
            )

            neighbor_stds.append(
                np.nan
            )

            healthy_counts.append(
                0
            )

            healthy_fractions.append(
                np.nan
            )

            neighborhood_ratios.append(
                np.nan
            )

            robustness_scores.append(
                np.nan
            )

            continue

        # -----------------------------------------------------
        # Descriptive neighborhood statistics
        # -----------------------------------------------------

        neighbor_mean = float(
            valid_neighbor_metrics.mean()
        )

        neighbor_median = float(
            valid_neighbor_metrics.median()
        )

        neighbor_minimum = float(
            valid_neighbor_metrics.min()
        )

        neighbor_std = float(
            valid_neighbor_metrics.std(
                ddof=0
            )
        )

        neighbor_means.append(
            neighbor_mean
        )

        neighbor_medians.append(
            neighbor_median
        )

        neighbor_minimums.append(
            neighbor_minimum
        )

        neighbor_stds.append(
            neighbor_std
        )

        # -----------------------------------------------------
        # Healthy-neighbor fraction
        # -----------------------------------------------------

        healthy_threshold = (
            center_metric
            * config.relative_performance_floor
        )

        healthy_neighbor_count = int(
            (
                valid_neighbor_metrics
                >= healthy_threshold
            ).sum()
        )

        healthy_fraction = (
            healthy_neighbor_count
            / valid_neighbor_count
        )

        healthy_counts.append(
            healthy_neighbor_count
        )

        healthy_fractions.append(
            healthy_fraction
        )

        # -----------------------------------------------------
        # Neighborhood performance ratio
        # -----------------------------------------------------

        neighborhood_ratio = (
            neighbor_median
            / center_metric
        )

        neighborhood_ratios.append(
            neighborhood_ratio
        )

        # -----------------------------------------------------
        # Combined robustness
        # -----------------------------------------------------

        clipped_ratio = min(
            max(
                neighborhood_ratio,
                0.0,
            ),
            1.0,
        )

        robustness = (
            healthy_fraction
            * clipped_ratio
        )

        robustness_scores.append(
            robustness
        )

    # ---------------------------------------------------------
    # Attach robustness results
    # ---------------------------------------------------------

    output[
        "neighbor_count"
    ] = neighbor_counts

    output[
        "valid_neighbor_count"
    ] = valid_neighbor_counts

    output[
        "neighbor_metric_mean"
    ] = neighbor_means

    output[
        "neighbor_metric_median"
    ] = neighbor_medians

    output[
        "neighbor_metric_min"
    ] = neighbor_minimums

    output[
        "neighbor_metric_std"
    ] = neighbor_stds

    output[
        "healthy_neighbor_count"
    ] = healthy_counts

    output[
        "healthy_neighbor_fraction"
    ] = healthy_fractions

    output[
        "neighborhood_ratio"
    ] = neighborhood_ratios

    output[
        "neighborhood_robustness"
    ] = robustness_scores

    return output