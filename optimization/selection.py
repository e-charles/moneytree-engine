# /optimization/selection.py

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from optimization.search_space import (
    DoubleDipParameters,
)


SUPPORTED_RANKING_METRICS = {
    "sharpe",
    "annualized_return",
    "total_return",
    "profit_factor",
}


@dataclass(frozen=True)
class SelectionConfig:
    """
    Rules used to determine which optimization candidates
    are eligible and how eligible candidates are ranked.

    min_closed_trades:
        Minimum number of completed trades required before
        a candidate may be considered.

    ranking_metric:
        Metric used to rank eligible candidates.

    max_allowed_drawdown:
        Optional maximum acceptable drawdown magnitude.

        Example:
            0.30 means candidates with drawdowns worse than
            -30% are rejected.

        None disables this filter.

    require_positive_return:
        If True, candidates with total_return <= 0 are rejected.
    """

    min_closed_trades: int = 10

    ranking_metric: str = "sharpe"

    max_allowed_drawdown: float | None = None

    require_positive_return: bool = True

    def __post_init__(self) -> None:

        if self.min_closed_trades < 1:
            raise ValueError(
                "min_closed_trades must be at least 1"
            )

        if self.ranking_metric not in SUPPORTED_RANKING_METRICS:
            raise ValueError(
                "Unsupported ranking_metric: "
                f"{self.ranking_metric}"
            )

        if self.max_allowed_drawdown is not None:
            if not 0 < self.max_allowed_drawdown <= 1:
                raise ValueError(
                    "max_allowed_drawdown must be "
                    "greater than 0 and less than or equal to 1"
                )


@dataclass(frozen=True)
class SelectedDoubleDipCandidate:
    """
    Result of candidate selection.
    """

    parameters: DoubleDipParameters

    ranking_metric: str
    ranking_value: float

    closed_trades: int
    total_return: float
    annualized_return: float
    max_drawdown: float
    sharpe: float
    profit_factor: float


def _validate_results(
    results: pd.DataFrame,
) -> None:
    """
    Validate the raw static optimizer result table.
    """

    if results.empty:
        raise ValueError(
            "optimization results cannot be empty"
        )

    required_columns = {
        "rsi_length",
        "oversold",
        "overbought",
        "max_bars_between_dips",
        "closed_trades",
        "total_return",
        "annualized_return",
        "max_drawdown",
        "sharpe",
        "profit_factor",
    }

    missing = required_columns.difference(
        results.columns
    )

    if missing:
        raise ValueError(
            "Missing required optimization columns: "
            f"{sorted(missing)}"
        )


def filter_eligible_candidates(
    results: pd.DataFrame,
    config: SelectionConfig,
) -> pd.DataFrame:
    """
    Apply candidate eligibility rules.

    Returns a new DataFrame.

    The original optimizer result table is not modified.
    """

    _validate_results(results)

    eligible = results.copy()

    # ---------------------------------------------------------
    # 1. Sufficient number of completed trades
    # ---------------------------------------------------------

    eligible = eligible[
        eligible["closed_trades"]
        >= config.min_closed_trades
    ]

    # ---------------------------------------------------------
    # 2. Positive return requirement
    # ---------------------------------------------------------

    if config.require_positive_return:
        eligible = eligible[
            eligible["total_return"] > 0
        ]

    # ---------------------------------------------------------
    # 3. Drawdown constraint
    #
    # max_drawdown is stored as a negative number.
    #
    # If:
    #
    #     max_allowed_drawdown = 0.30
    #
    # Then:
    #
    #     -0.10 -> allowed
    #     -0.20 -> allowed
    #     -0.30 -> allowed
    #     -0.31 -> rejected
    #     -0.50 -> rejected
    # ---------------------------------------------------------

    if config.max_allowed_drawdown is not None:
        eligible = eligible[
            eligible["max_drawdown"]
            >= -config.max_allowed_drawdown
        ]

    # ---------------------------------------------------------
    # 4. Ranking metric must exist
    # ---------------------------------------------------------
    #
    # Examples:
    #
    # A candidate can have Sharpe = NaN when volatility
    # is zero or there is insufficient return data.
    #
    # Such a candidate cannot participate in a Sharpe-based
    # ranking.
    #
    # Note:
    # profit_factor = inf is intentionally allowed.
    # That is a valid metric result when there are no losing
    # trades.
    # ---------------------------------------------------------

    metric = config.ranking_metric

    eligible = eligible[
        eligible[metric].notna()
    ]

    return eligible.copy()


def rank_candidates(
    results: pd.DataFrame,
    config: SelectionConfig,
) -> pd.DataFrame:
    """
    Filter eligible candidates and rank them.

    The primary ranking criterion is the configured ranking
    metric.

    Tie breakers:

        1. More closed trades
        2. Higher total return
        3. Smaller maximum drawdown

    Returns a new DataFrame sorted from best to worst.
    """

    eligible = filter_eligible_candidates(
        results=results,
        config=config,
    )

    if eligible.empty:
        return eligible

    metric = config.ranking_metric

    # ---------------------------------------------------------
    # Avoid adding the primary metric twice.
    #
    # For example, if ranking_metric == "total_return",
    # we do not want:
    #
    # sort by:
    # total_return,
    # closed_trades,
    # total_return,
    # max_drawdown
    # ---------------------------------------------------------

    sort_columns = [metric]
    ascending = [False]

    if "closed_trades" != metric:
        sort_columns.append(
            "closed_trades"
        )
        ascending.append(False)

    if "total_return" != metric:
        sort_columns.append(
            "total_return"
        )
        ascending.append(False)

    if "max_drawdown" != metric:
        sort_columns.append(
            "max_drawdown"
        )

        # max_drawdown is negative.
        #
        # -0.10 is better than -0.30.
        #
        # Therefore higher values are better.
        ascending.append(False)

    ranked = eligible.sort_values(
        by=sort_columns,
        ascending=ascending,

        # Stable sorting is useful when candidates remain
        # completely tied.
        kind="mergesort",
    )

    return ranked.reset_index(
        drop=True
    )


def select_double_dip_candidate(
    results: pd.DataFrame,
    config: SelectionConfig | None = None,
) -> SelectedDoubleDipCandidate:
    """
    Select the highest-ranked eligible Double Dip candidate.

    The supplied results should contain ONLY information
    available during the optimizer's in-sample period.

    During walk-forward optimization the selected parameters
    are then frozen and used on the following unseen period.
    """

    if config is None:
        config = SelectionConfig()

    ranked = rank_candidates(
        results=results,
        config=config,
    )

    if ranked.empty:
        raise ValueError(
            "No eligible optimization candidates "
            "satisfied the selection rules"
        )

    winner = ranked.iloc[0]

    # ---------------------------------------------------------
    # Reconstruct strongly typed strategy parameters
    # ---------------------------------------------------------

    parameters = DoubleDipParameters(
        rsi_length=int(
            winner["rsi_length"]
        ),
        oversold=float(
            winner["oversold"]
        ),
        overbought=float(
            winner["overbought"]
        ),
        max_bars_between_dips=int(
            winner["max_bars_between_dips"]
        ),
    )

    # ---------------------------------------------------------
    # Return selected candidate plus useful in-sample metrics
    # ---------------------------------------------------------

    return SelectedDoubleDipCandidate(
        parameters=parameters,

        ranking_metric=config.ranking_metric,

        ranking_value=float(
            winner[
                config.ranking_metric
            ]
        ),

        closed_trades=int(
            winner["closed_trades"]
        ),

        total_return=float(
            winner["total_return"]
        ),

        annualized_return=float(
            winner["annualized_return"]
        ),

        max_drawdown=float(
            winner["max_drawdown"]
        ),

        sharpe=float(
            winner["sharpe"]
        ),

        profit_factor=float(
            winner["profit_factor"]
        ),
    )