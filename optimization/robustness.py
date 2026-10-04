# /optimization/robustness.py

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


PARAMETER_COLUMNS = (
    "rsi_length",
    "oversold",
    "overbought",
    "max_bars_between_dips",
)


@dataclass(frozen=True)
class NeighborhoodConfig:
    """
    Configuration controlling parameter-neighborhood robustness.

    metric:
        Performance metric used to evaluate the neighborhood.

        Sharpe is the recommended initial choice because this
        robustness layer is intended to measure whether nearby
        parameter configurations have reasonably similar
        risk-adjusted behavior.

    min_neighbors:
        Minimum number of neighboring configurations required
        before neighborhood robustness is considered valid.

    relative_performance_floor:
        A neighbor is considered "healthy" when its metric is at
        least this fraction of the center candidate's metric.

        Example:
            center Sharpe = 1.50
            floor = 0.75

            healthy neighbor threshold = 1.125

    include_diagonal_neighbors:
        If False, only candidates differing in exactly one
        parameter dimension are considered neighbors.

        If True, candidates may differ by one grid step across
        multiple parameter dimensions simultaneously.

    require_positive_neighbor_metric:
        Prevents negative-performance neighbors from contributing
        positively to robustness.
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

        if not 0 < self.relative_performance_floor <= 1:
            raise ValueError(
                "relative_performance_floor must be "
                "greater than 0 and less than or equal to 1"
            )


def _validate_results(
    results: pd.DataFrame,
    config: NeighborhoodConfig,
) -> None:
    """
    Validate static optimizer results before neighborhood analysis.
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
            "opt*mization results contain duplicate*"
            "parameter configura*ions"
        )


def _ordered_parameter_values(
    results: pd.DataFrame,
) -> dict[str, list[float | int]]:
    """
    Extract the ordered parameter grid represented by the
    optimizer result table.
    """

    return {
        column: sorted(
            results[column].dropna().unique().tolist()
        )
        for column in PARAMETER_COLUMNS
    }


def _parameter_grid_positions(
    values: dict[str, list[float | int]],
) -> dict[str, dict[float | int, int]]:
    """
    Map each parameter value to its location on the search grid.

    Example:

        rsi_length:
            5  -> position 0
            8  -> position 1
            14 -> position 2

    Neighborhood distance is based on grid position rather than
    numeric distance.

    This matters because search spaces do not necessarily use
    evenly spaced parameter values.
    """

    return {
        column: {
            value: position
            for position, value in enumerate(column_values)
        }
        for column, column_values in values.items()
    }


def _grid_distance(
    candidate: pd.Series,
    neighbor: pd.Series,
    positions: dict[str, dict[float | int, int]],
) -> tuple[int, int]:
    """
    Calculate grid distance between two parameter configurations.

    Returns:
        (maximum_axis_distance, changed_dimensions)

    maximum_axis_distance:
        Largest number of grid steps separating the candidates
        along any one parameter.

    changed_dimensions:
        Number of parameter dimensions that differ.
    """

    distances = []

    for column in PARAMETER_COLUMNS:
        candidate_position = positions[column][
            candidate[column]
        ]

        neighbor_position = positions[column][
            neighbor[column]
        ]

        distances.append(
            abs(
                candidate_position
                - neighbor_position
            )
        )

    maximum_axis_distance = max(distances)

    changed_dimensions = sum(
        distance > 0
        for distance in distances
    )

    return (
        maximum_axis_distance,
        changed_dimensions,
    )


def _find_neighbor_indices(
    results: pd.DataFrame,
    row_index: int,
    positions: dict[str, dict[float | int, int]],
    include_diagonal_neighbors: bool,
) -> list[int]:
    """
    Find immediate neighbors of one candidate.

    A direct neighbor differs by exactly one search-grid step.

    By default, only one parameter may change at a time.

    Example direct neighbors:

        RSI 12 -> RSI 11
        RSI 12 -> RSI 13

        oversold 30 -> 25
        oversold 30 -> 35

    Diagonal neighbors may differ by one grid step in more than
    one parameter simultaneously.
    """

    candidate = results.loc[row_index]

    neighbor_indices: list[int] = []

    for other_index, other in results.iterrows():
        if other_index == row_index:
            continue

        (
            maximum_axis_distance,
            changed_dimensions,
        ) = _grid_distance(
            candidate=candidate,
            neighbor=other,
            positions=positions,
        )

        # Must be one immediate grid step away.
        if maximum_axis_distance != 1:
            continue

        if include_diagonal_neighbors:
            if changed_dimensions >= 1:
                neighbor_indices.append(other_index)

        else:
            # Direct axis-aligned neighbor.
            if changed_dimensions == 1:
                neighbor_indices.append(other_index)

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
        Number of immediate parameter-grid neighbors.

    valid_neighbor_count:
        Number of neighbors with a finite, usable metric.

    neighbor_metric_mean:
        Average metric across valid neighbors.

    neighbor_metric_median:
        Median metric across valid neighbors.

    neighbor_metric_min:
        Worst valid neighboring metric.

    neighbor_metric_std:
        Standard deviation of neighboring performance.

    healthy_neighbor_count:
        Number of neighbors achieving at least
        relative_performance_floor * center metric.

    healthy_neighbor_fraction:
        Fraction of valid neighbors considered healthy.

    neighborhood_ratio:
        Median neighboring performance divided by center
        candidate performance.

        Values near 1 indicate nearby configurations perform
        comparably to the center.

    neighborhood_robustness:
        Combined robustness statistic.

        For positive center metrics:

            healthy_neighbor_fraction
            *
            clipped neighborhood_ratio

        Scores lie between 0 and 1.

        A candidate receives NaN if insufficient neighbors exist.
    """

    if config is None:
        config = NeighborhoodConfig()

    _validate_results(
        results=results,
        config=config,
    )

    output = results.copy().reset_index(
        drop=True
    )

    values = _ordered_parameter_values(
        output
    )

    positions = _parameter_grid_positions(
        values
    )

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

    for row_index in output.index:

        center_metric = float(
            output.loc[
                row_index,
                metric,
            ]
        )

        neighbor_indices = _find_neighbor_indices(
            results=output,
            row_index=row_index,
            positions=positions,
            include_diagonal_neighbors=(
                config.include_diagonal_neighbors
            ),
        )

        neighbor_count = len(
            neighbor_indices
        )

        neighbor_counts.append(
            neighbor_count
        )

        neighbor_metrics = output.loc[
            neighbor_indices,
            metric,
        ].astype(float)

        # Remove NaN and infinity.
        valid_mask = np.isfinite(
            neighbor_metrics.to_numpy()
        )

        valid_neighbor_metrics = (
            neighbor_metrics.iloc[
                np.flatnonzero(valid_mask)
            ]
        )

        if config.require_positive_neighbor_metric:
            valid_neighbor_metrics = (
                valid_neighbor_metrics[
                    valid_neighbor_metrics > 0
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
            or not np.isfinite(center_metric)
            or center_metric <= 0
        ):
            neighbor_means.append(np.nan)
            neighbor_medians.append(np.nan)
            neighbor_minimums.append(np.nan)
            neighbor_stds.append(np.nan)

            healthy_counts.append(0)
            healthy_fractions.append(np.nan)

            neighborhood_ratios.append(np.nan)
            robustness_scores.append(np.nan)

            continue

        # -----------------------------------------------------
        # Neighborhood descriptive statistics
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
        # Healthy-neighbor calculation
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
        # Neighborhood ratio
        # -----------------------------------------------------
        #
        # center = 1.50
        # median neighbor = 1.35
        #
        # ratio = 0.90
        #
        # A ratio near 1 means the selected configuration is not
        # dramatically better than its immediate surroundings.
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
        #
        # Cap neighborhood ratio at 1.
        #
        # If neighboring configurations happen to outperform
        # the center candidate, that should not create a
        # robustness score greater than 1.
        # -----------------------------------------------------

        clipped_ratio = min(
            max(neighborhood_ratio, 0.0),
            1.0,
        )

        robustness = (
            healthy_fraction
            * clipped_ratio
        )

        robustness_scores.append(
            robustness
        )

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