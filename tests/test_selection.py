import numpy as np
import pandas as pd
import pytest

from optimization.search_space import (
    DoubleDipParameters,
)

from optimization.selection import (
    SelectionConfig,
    filter_eligible_candidates,
    rank_candidates,
    select_double_dip_candidate,
)


# ============================================================
# HELPERS
# ============================================================


def make_results() -> pd.DataFrame:
    """
    Create deterministic static-optimizer results.

    Candidate A:
        Highest Sharpe, but only 3 trades.
        Should normally be rejected.

    Candidate B:
        Best eligible Sharpe.
        Should normally win.

    Candidate C:
        Valid, but lower Sharpe than B.

    Candidate D:
        Negative return.
        Should be rejected when positive return is required.

    Candidate E:
        NaN Sharpe.
        Cannot participate in Sharpe ranking.
    """

    return pd.DataFrame(
        {
            "rsi_length": [ 7, 12, 14, 18, 20],
            "oversold": [ 25.0, 30.0, 30.0, 25.0, 30.0,],
            "overbought": [ 75.0, 70.0, 70.0, 75.0, 70.0,],
            "max_bars_between_dips": [ 5, 7, 10, 5, 10,],
            "closed_trades": [3, 22, 30, 18, 20,],
            "total_return": [0.60, 0.35, 0.29, -0.05, 0.20,],
            "annualized_return": [0.45, 0.22, 0.18, -0.03, 0.12,],
            "max_drawdown": [-0.08, -0.14, -0.10, -0.20, -0.05,],
            "sharpe": [3.20, 1.60, 1.40, 0.50, np.nan,],
            "profit_factor": [ np.inf, 2.10, 1.80, 0.80, 1.50,],
        }
    )


# ============================================================
# ELIGIBILITY
# ============================================================


def test_low_trade_candidate_is_rejected() -> None:
    results = make_results()

    config = SelectionConfig(
        min_closed_trades=10,
        ranking_metric="sharpe",
    )

    eligible = filter_eligible_candidates(
        results,
        config,
    )

    assert 7 not in eligible["rsi_length"].tolist()

    assert 12 in eligible["rsi_length"].tolist()


def test_negative_return_candidate_is_rejected() -> None:
    results = make_results()

    config = SelectionConfig(
        min_closed_trades=10,
        ranking_metric="sharpe",
        require_positive_return=True,
    )

    eligible = filter_eligible_candidates(
        results,
        config,
    )

    assert 18 not in eligible["rsi_length"].tolist()


def test_negative_return_allowed_when_not_required() -> None:
    results = make_results()

    config = SelectionConfig(
        min_closed_trades=10,
        ranking_metric="sharpe",
        require_positive_return=False,
    )

    eligible = filter_eligible_candidates(
        results,
        config,
    )

    assert 18 in eligible["rsi_length"].tolist()


def test_excessive_drawdown_candidate_is_rejected() -> None:
    results = make_results()

    config = SelectionConfig(
        min_closed_trades=10,
        ranking_metric="sharpe",
        max_allowed_drawdown=0.12,
    )

    eligible = filter_eligible_candidates(
        results,
        config,
    )

    # RSI 12 has drawdown = -14%.
    assert 12 not in eligible["rsi_length"].tolist()

    # RSI 14 has drawdown = -10%.
    assert 14 in eligible["rsi_length"].tolist()


def test_drawdown_exactly_at_limit_is_allowed() -> None:
    results = make_results()

    # Change RSI 14 to exactly -10%.
    config = SelectionConfig(
        min_closed_trades=10,
        ranking_metric="sharpe",
        max_allowed_drawdown=0.10,
    )

    eligible = filter_eligible_candidates(
        results,
        config,
    )

    assert 14 in eligible["rsi_length"].tolist()


def test_nan_ranking_metric_is_rejected() -> None:
    results = make_results()

    config = SelectionConfig(
        min_closed_trades=10,
        ranking_metric="sharpe",
    )

    eligible = filter_eligible_candidates(
        results,
        config,
    )

    # RSI 20 has Sharpe = NaN.
    assert 20 not in eligible["rsi_length"].tolist()


# ============================================================
# RANKING
# ============================================================


def test_highest_eligible_sharpe_is_ranked_first() -> None:
    results = make_results()

    config = SelectionConfig(
        min_closed_trades=10,
        ranking_metric="sharpe",
    )

    ranked = rank_candidates(
        results,
        config,
    )

    assert ranked.iloc[0]["rsi_length"] == 12

    assert ranked.iloc[0]["sharpe"] == pytest.approx(
        1.60
    )


# def test_more_trades_break_exact_metric_tie() -> None:
#     results = pd.DataFrame(





def test_infinite_profit_factor_can_be_ranked() -> None:
    results = pd.DataFrame(
        {
            "rsi_length": [10, 14],
            "oversold": [30.0, 30.0],
            "overbought": [70.0, 70.0],
            "max_bars_between_dips": [5, 5],

            "closed_trades": [20, 25],

            "total_return": [0.20, 0.25],
            "annualized_return": [0.10, 0.12],

            "max_drawdown": [-0.10, -0.10],

            "sharpe": [1.0, 1.1],

            "profit_factor": [
                2.0,
                np.inf,
            ],
        }
    )

    config = SelectionConfig(
        min_closed_trades=10,
        ranking_metric="profit_factor",
    )

    ranked = rank_candidates(
        results,
        config,
    )

    assert ranked.iloc[0]["rsi_length"] == 14

    assert np.isinf(
        ranked.iloc[0]["profit_factor"]
    )