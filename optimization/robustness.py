# /optimization/robustness.py

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd
from numpy.typing import NDArray


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

FloatArray = NDArray[np.float64]


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
        A neighbor is healthy when its metric is at least this
        fraction of the center candidate's metric.

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

    Validation occurs once before entering the NumPy-backed
    robustness calculation.
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
    Extract ordered values for every parameter dimension.

    Adjacency is determined by grid position rather than
    numerical distance.

    Example:

        RSI values:

            5, 8, 14, 21

        Grid positions:

            0, 1, 2, 3

    Therefore 8 and 14 are immediate neighbors even though
    their numerical difference is 6.
    """

    return {
        column: sorted(
            results[
                column
            ]
            .dropna()
            .unique()
            .tolist()
        )
        for column in PARAMETER_COLUMNS
    }


def _parameter_grid_positions(
    values: dict[str, list[ParameterValue]],
) -> dict[
    str,
    dict[ParameterValue, int],
]:
    """
    Map every parameter value to its position along its grid
    dimension.
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


def _build_candidate_keys(
    results: pd.DataFrame,
) -> list:
    """
    Convert optimizer parameter columns into lightweight Python
    tuples once.

    This prevents repeated DataFrame row extraction inside the
    robustness loop.
    """

    parameter_frame = results[
        list(PARAMETER_COLUMNS)
    ]

    return [
        (
            row.rsi_length,
            row.oversold,
            row.overbought,
            row.max_bars_between_dips,
        )
        for row in parameter_frame.itertuples(
            index=False
        )
    ]


def _build_candidate_lookup(
    candidate_keys: list[ParameterKey],
) -> dict[ParameterKey, int]:
    """
    Build a direct mapping:

        parameter tuple -> result row

    Example:

        (
            14,
            30.0,
            70.0,
            5,
        )

        ->

        1024

    Neighbor discovery can therefore use constant-time
    dictionary lookups rather than DataFrame searches.
    """

    return {
        key: index
        for index, key
        in enumerate(candidate_keys)
    }


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

        grid:
            [5, 8, 14, 21]

        center:
            14

        result:
            [14, 8, 21]
    """

    axis_values = values[
        column
    ]

    center_position = positions[
        column
    ][value]

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

    if next_position < len(
        axis_values
    ):

        choices.append(
            axis_values[
                next_position
            ]
        )

    return choices


def _find_neighbor_indices(
    center_key: ParameterKey,
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

    No pandas objects are used.

    Direct mode:
        Exactly one parameter differs by one grid step.

    Diagonal mode:
        One or more parameters may differ by one grid step,
        while no parameter differs by more than one step.
    """

    # =========================================================
    # DIRECT AXIS-ALIGNED NEIGHBORS
    # =========================================================
    #
    # Four dimensions means at most:
    #
    #     4 dimensions × 2 directions
    #     = 8 theoretical neighbors
    # =========================================================

    if not include_diagonal_neighbors:

        neighbor_indices: list[int] = []

        for dimension, column in enumerate(
            PARAMETER_COLUMNS
        ):

            current_value = (
                center_key[
                    dimension
                ]
            )

            current_position = (
                positions[
                    column
                ][current_value]
            )

            axis_values = (
                values[
                    column
                ]
            )

            # -----------------------------------------------
            # Previous / next grid position
            # -----------------------------------------------

            for offset in (-1, 1):

                neighbor_position = (
                    current_position
                    + offset
                )

                if not (
                    0
                    <= neighbor_position
                    < len(axis_values)
                ):
                    continue

                # Construct the neighbor key without touching
                # the optimization DataFrame.

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

                # Some theoretical neighbors may not exist.
                #
                # For example, the search space excludes:
                #
                #     oversold >= overbought
                #
                if neighbor_index is not None:

                    neighbor_indices.append(
                        neighbor_index
                    )

        return neighbor_indices

    # =========================================================
    # DIAGONAL NEIGHBORS
    # =========================================================
    #
    # Each dimension contributes:
    #
    #     previous
    #     center
    #     next
    #
    # With four dimensions there can be at most:
    #
    #     3^4 - 1 = 80
    #
    # neighborhood combinations.
    # =========================================================

    axis_choices = [
        _neighbor_axis_values(
            column=column,
            value=center_key[dimension],
            values=values,
            positions=positions,
        )
        for dimension, column
        in enumerate(PARAMETER_COLUMNS)
    ]

    neighbor_indices: list[int] = []

    for combination in product(
        *axis_choices
    ):

        neighbor_key: ParameterKey = (
            combination[0],
            combination[1],
            combination[2],
            combination[3],
        )

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

    The DataFrame is used only as the external interface.

    Internally:

        DataFrame
            |
            v
        validate once
            |
            v
        metric -> NumPy
        parameters -> tuples
            |
            v
        NumPy / Python robustness loop
            |
            v
        result NumPy arrays
            |
            v
        attach columns to output DataFrame

    The original DataFrame is not modified.
    """

    if config is None:
        config = NeighborhoodConfig()

    # =========================================================
    # 1. VALIDATE ONCE
    # =========================================================

    _validate_results(
        results=results,
        config=config,
    )

    output = (
        results
        .copy()
        .reset_index(
            drop=True
        )
    )

    candidate_count = len(
        output
    )

    metric = config.metric

    # =========================================================
    # 2. EXTRACT METRIC TO NUMPY ONCE
    # =========================================================
    #
    # This removes:
    #
    #     output.loc[...]
    #     .astype(float)
    #     .iloc[...]
    #     Series.mean()
    #     Series.median()
    #     Series.std()
    #
    # from the candidate loop.
    # =========================================================

    metric_values: FloatArray = (
        output[
            metric
        ].to_numpy(
            dtype=np.float64,
            copy=False,
        )
    )

    # =========================================================
    # 3. BUILD PARAMETER GRID STRUCTURES
    # =========================================================

    values = (
        _ordered_parameter_values(
            output
        )
    )

    positions = (
        _parameter_grid_positions(
            values
        )
    )

    # =========================================================
    # 4. BUILD PARAMETER KEYS ONCE
    # =========================================================

    candidate_keys = (
        _build_candidate_keys(
            output
        )
    )

    candidate_lookup = (
        _build_candidate_lookup(
            candidate_keys
        )
    )

    # =========================================================
    # 5. PREALLOCATE OUTPUT ARRAYS
    # =========================================================
    #
    # We know exactly how many candidates exist, so no Python
    # result lists need to grow dynamically.
    # =========================================================

    neighbor_counts = np.empty(
        candidate_count,
        dtype=np.int64,
    )

    valid_neighbor_counts = np.empty(
        candidate_count,
        dtype=np.int64,
    )

    neighbor_means = np.full(
        candidate_count,
        np.nan,
        dtype=np.float64,
    )

    neighbor_medians = np.full(
        candidate_count,
        np.nan,
        dtype=np.float64,
    )

    neighbor_minimums = np.full(
        candidate_count,
        np.nan,
        dtype=np.float64,
    )

    neighbor_stds = np.full(
        candidate_count,
        np.nan,
        dtype=np.float64,
    )

    healthy_counts = np.zeros(
        candidate_count,
        dtype=np.int64,
    )

    healthy_fractions = np.full(
        candidate_count,
        np.nan,
        dtype=np.float64,
    )

    neighborhood_ratios = np.full(
        candidate_count,
        np.nan,
        dtype=np.float64,
    )

    robustness_scores = np.full(
        candidate_count,
        np.nan,
        dtype=np.float64,
    )

    # =========================================================
    # 6. EVALUATE EACH CANDIDATE
    # =========================================================

    for row_index in range(
        candidate_count
    ):

        center_key = (
            candidate_keys[
                row_index
            ]
        )

        center_metric = float(
            metric_values[
                row_index
            ]
        )

        # -----------------------------------------------------
        # Neighbor lookup
        # -----------------------------------------------------

        neighbor_indices = (
            _find_neighbor_indices(
                center_key=center_key,
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

        neighbor_counts[
            row_index
        ] = neighbor_count

        # -----------------------------------------------------
        # No neighbors
        # -----------------------------------------------------

        if neighbor_count == 0:

            valid_neighbor_counts[
                row_index
            ] = 0

            continue

        # -----------------------------------------------------
        # Direct NumPy metric lookup
        # -----------------------------------------------------

        neighbor_metrics = (
            metric_values[
                neighbor_indices
            ]
        )

        # -----------------------------------------------------
        # Remove NaN / +/- infinity
        # -----------------------------------------------------

        valid_neighbor_metrics = (
            neighbor_metrics[
                np.isfinite(
                    neighbor_metrics
                )
            ]
        )

        # -----------------------------------------------------
        # Optional positive-performance requirement
        # -----------------------------------------------------

        if (
            config
            .require_positive_neighbor_metric
        ):

            valid_neighbor_metrics = (
                valid_neighbor_metrics[
                    valid_neighbor_metrics
                    > 0.0
                ]
            )

        valid_neighbor_count = (
            valid_neighbor_metrics.size
        )

        valid_neighbor_counts[
            row_index
        ] = (
            valid_neighbor_count
        )

        # =====================================================
        # INSUFFICIENT INFORMATION
        # =====================================================

        if (
            valid_neighbor_count
            < config.min_neighbors
            or not np.isfinite(
                center_metric
            )
            or center_metric <= 0.0
        ):
            # Output arrays were initialized to:
            #
            #     statistics -> NaN
            #     healthy count -> 0
            #
            # so nothing else needs to be assigned here.

            continue

        # =====================================================
        # NEIGHBORHOOD STATISTICS
        # =====================================================

        neighbor_mean = float(
            np.mean(
                valid_neighbor_metrics
            )
        )

        neighbor_median = float(
            np.median(
                valid_neighbor_metrics
            )
        )

        neighbor_minimum = float(
            np.min(
                valid_neighbor_metrics
            )
        )

        neighbor_std = float(
            np.std(
                valid_neighbor_metrics,
                ddof=0,
            )
        )

        neighbor_means[
            row_index
        ] = neighbor_mean

        neighbor_medians[
            row_index
        ] = neighbor_median

        neighbor_minimums[
            row_index
        ] = neighbor_minimum

        neighbor_stds[
            row_index
        ] = neighbor_std

        # =====================================================
        # HEALTHY NEIGHBORS
        # =====================================================

        healthy_threshold = (
            center_metric
            * config.relative_performance_floor
        )

        healthy_neighbor_count = int(
            np.count_nonzero(
                valid_neighbor_metrics
                >= healthy_threshold
            )
        )

        healthy_fraction = (
            healthy_neighbor_count
            / valid_neighbor_count
        )

        healthy_counts[
            row_index
        ] = (
            healthy_neighbor_count
        )

        healthy_fractions[
            row_index
        ] = (
            healthy_fraction
        )

        # =====================================================
        # NEIGHBORHOOD RATIO
        # =====================================================

        neighborhood_ratio = (
            neighbor_median
            / center_metric
        )

        neighborhood_ratios[
            row_index
        ] = (
            neighborhood_ratio
        )

        # =====================================================
        # COMBINED ROBUSTNESS
        # =====================================================
        #
        # Equivalent to:
        #
        #     min(
        #         max(ratio, 0),
        #         1,
        #     )
        #
        # but NumPy expresses the intent directly.
        # =====================================================

        clipped_ratio = float(
            np.clip(
                neighborhood_ratio,
                0.0,
                1.0,
            )
        )

        robustness_scores[
            row_index
        ] = (
            healthy_fraction
            * clipped_ratio
        )

    # =========================================================
    # 7. ATTACH RESULT ARRAYS TO DATAFRAME ONCE
    # =========================================================

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