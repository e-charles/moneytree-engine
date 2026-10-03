# /optimization/static.py

from __future__ import annotations

import numpy as np
import pandas as pd

from ta.momentum import RSIIndicator

from strats.double_dip import rsi_double_dip_targets

from backtesting.engine import BacktestConfig, run_backtest

from backtesting.metrics import calculate_metrics

from optimization.search_space import DoubleDipParameters, DoubleDipSearchSpace


def _validate_market_data(data: pd.DataFrame) -> None:
    """
    Validate the market data required by the optimizer.
    """

    if data.empty:
        raise ValueError("data cannot be empty")

    required_columns = {
        "Open",
        "Close",
    }

    missing = required_columns.difference(
        data.columns
    )

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    if data.index.has_duplicates:
        raise ValueError(
            "data index cannot contain duplicate timestamps"
        )

    if not data.index.is_monotonic_increasing:
        raise ValueError(
            "data index must be sorted in ascending order"
        )

    if data[["Open", "Close"]].isna().any().any():
        raise ValueError(
            "Open and Close cannot contain NaN values"
        )

    if (data[["Open", "Close"]] <= 0).any().any():
        raise ValueError(
            "Open and Close prices must be greater than zero"
        )


def _calculate_rsi(
    close: pd.Series,
    rsi_length: int,
) -> pd.Series:
    """
    Calculate RSI for one candidate configuration.
    """

    return RSIIndicator(
        close=close,
        window=rsi_length,
    ).rsi()


def evaluate_double_dip_candidate_from_rsi(
    data: pd.DataFrame,
    rsi: pd.Series,
    parameters: DoubleDipParameters,
    backtest_config: BacktestConfig | None = None,
    periods_per_year: int = 252,
) -> dict[str, float | int]:
    """
    Evaluate one Double Dip parameter configuration using an already-calculated RSI series

    Returns one flat dictionary suitable for conversion
    into a DataFrame row.
    """

    if backtest_config is None:
        backtest_config = BacktestConfig()

    # ---------------------------------------------------------
    # Strategy
    # ---------------------------------------------------------

    target_at_close = rsi_double_dip_targets(
        rsi=rsi,
        oversold=parameters.oversold,
        overbought=parameters.overbought,
        max_bars_between_dips=(
            parameters.max_bars_between_dips
        ),
    )

    # ---------------------------------------------------------
    # Backtest
    # ---------------------------------------------------------

    result = run_backtest(
        data=data,
        target_at_close=target_at_close,
        config=backtest_config,
    )

    # ---------------------------------------------------------
    # Metrics
    # ---------------------------------------------------------

    metrics = calculate_metrics(
        equity=result.equity,
        trades=result.trades,
        periods_per_year=periods_per_year,
    )

    # ---------------------------------------------------------
    # Flatten result
    # ---------------------------------------------------------

    return {
        "rsi_length": parameters.rsi_length,
        "oversold": parameters.oversold,
        "overbought": parameters.overbought,
        "max_bars_between_dips": (
            parameters.max_bars_between_dips
        ),
        **metrics, # should destructire metrics
        "final_equity": float(
        result.equity.iloc[-1]
        ),
    }


def optimize_double_dip(
    data: pd.DataFrame,
    search_space: DoubleDipSearchSpace,
    backtest_config: BacktestConfig | None = None,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """
    Evaluate every valid RSI Double Dip candidate.

    IMPORTANT:
    This function does NOT select a winner.

    It returns the complete result table so that candidate
    eligibility, ranking, robustness, and walk-forward
    selection can be handled independently.
    """

    # ---------------------------------------------------------
    # 1. Validate market data
    # ---------------------------------------------------------

    _validate_market_data(data)

    if periods_per_year <= 0:
        raise ValueError(
            "periods_per_year must be greater than zero"
        )

    if backtest_config is None:
        backtest_config = BacktestConfig()

    # ---------------------------------------------------------
    # 2. Generate every valid candidate
    # ---------------------------------------------------------

    candidates = list(search_space.candidates())

    if not candidates:
        raise ValueError(
            "search_space produced no valid candidates"
        )

    # ---------------------------------------------------------
    # 3. Find the unique RSI lengths
    # ---------------------------------------------------------
    #
    # Many candidates share the same RSI length.
    #
    # Example:
    #
    # RSI 14 / OS 25 / OB 70 / Gap 5
    # RSI 14 / OS 30 / OB 70 / Gap 5
    # RSI 14 / OS 30 / OB 75 / Gap 10
    #
    # All three use exactly the same RSI(14) series.
    #
    # Therefore, calculate RSI(14) once rather than three times.
    # ---------------------------------------------------------

    unique_rsi_lengths = sorted(
        {
            candidate.rsi_length
            for candidate in candidates
        }
    )

    # ---------------------------------------------------------
    # 4. Build RSI cache
    # ---------------------------------------------------------

    rsi_cache: dict[int, pd.Series] = {}

    for rsi_length in unique_rsi_lengths:
        rsi_cache[rsi_length] = _calculate_rsi(
            close=data["Close"],
            rsi_length=rsi_length,
        )

    # ---------------------------------------------------------
    # 5. Evaluate every candidate
    # ---------------------------------------------------------

    rows: list[dict[str, float | int]] = []

    for parameters in candidates:

        # Retrieve the already-calculated RSI series.
        rsi = rsi_cache[
            parameters.rsi_length
        ]

        row = evaluate_double_dip_candidate_from_rsi(
            data=data,
            rsi=rsi,
            parameters=parameters,
            backtest_config=backtest_config,
            periods_per_year=periods_per_year,
        )

        rows.append(row)

    # ---------------------------------------------------------
    # 6. Convert all candidate results into a DataFrame
    # ---------------------------------------------------------

    results = pd.DataFrame(rows)

    # ---------------------------------------------------------
    # 7. Return RAW optimization results
    # ---------------------------------------------------------
    #
    # Do NOT:
    #
    # - sort by Sharpe
    # - pick highest return
    # - remove low-trade candidates
    # - replace infinite profit factors
    # - select a winner
    #
    # Those responsibilities belong to selection.py.
    # ---------------------------------------------------------

    return results