# /optimization/static.py

from __future__ import annotations

import numpy as np
import pandas as pd

from numpy.typing import NDArray
from ta.momentum import RSIIndicator

from strats.double_dip import (
    rsi_double_dip_targets,
)

from backtesting.engine import (
    BacktestConfig,
    run_backtest,
    validate_backtest_config,
)

from backtesting.metrics import (
    calculate_metrics,
)

from optimization.search_space import (
    DoubleDipParameters,
    DoubleDipSearchSpace,
)


FloatArray = NDArray[np.float64]


def _validate_market_data(
    data: pd.DataFrame,
) -> None:
    """
    Validate market data before entering the numerical
    optimization pipeline.

    Validation occurs once per optimization window.

    After validation, Open and Close are converted to NumPy
    arrays and reused by every candidate.
    """

    if data.empty:
        raise ValueError(
            "data cannot be empty"
        )

    required_columns = {
        "Open",
        "Close",
    }

    missing = required_columns.difference(
        data.columns
    )

    if missing:
        raise ValueError(
            "Missing required columns: "
            f"{sorted(missing)}"
        )

    if data.index.has_duplicates:
        raise ValueError(
            "data index cannot contain duplicate timestamps"
        )

    if not data.index.is_monotonic_increasing:
        raise ValueError(
            "data index must be sorted in ascending order"
        )

    prices = data[
        ["Open", "Close"]
    ]

    if prices.isna().any().any():
        raise ValueError(
            "Open and Close cannot contain NaN values"
        )

    if (
        prices <= 0
    ).any().any():
        raise ValueError(
            "Open and Close prices must be greater than zero"
        )


def _calculate_rsi(
    close: pd.Series,
    rsi_length: int,
) -> FloatArray:
    """
    Calculate one RSI series and immediately convert it to
    the optimizer's NumPy representation.

    The external RSI library currently operates on pandas,
    but the optimizer does not retain the resulting Series.
    """

    rsi = RSIIndicator(
        close=close,
        window=rsi_length,
    ).rsi()

    return rsi.to_numpy(
        dtype=np.float64,
        copy=False,
    )


def evaluate_double_dip_candidate_from_rsi(
    open_prices: FloatArray,
    close_prices: FloatArray,
    rsi: FloatArray,
    parameters: DoubleDipParameters,
    backtest_config: BacktestConfig,
    periods_per_year: int = 252,
) -> dict[str, float | int]:
    """
    Evaluate one Double Dip parameter configuration.

    This function is part of the optimizer's trusted numerical
    fast path.

    All market data has already been validated and converted
    to NumPy before this function is called.

    Returns one flat dictionary suitable for inclusion in the
    final optimization result DataFrame.
    """

    # =========================================================
    # 1. STRATEGY
    # =========================================================

    target_at_close = (
        rsi_double_dip_targets(
            rsi=rsi,
            oversold=parameters.oversold,
            overbought=parameters.overbought,
            max_bars_between_dips=(
                parameters.max_bars_between_dips
            ),
        )
    )

    # =========================================================
    # 2. BACKTEST
    # =========================================================

    result = run_backtest(
        open_prices=open_prices,
        close_prices=close_prices,
        target_at_close=target_at_close,
        config=backtest_config,

        # Market arrays, strategy output, and config are trusted
        # inside the optimizer candidate loop.
        inputs_prevalidated=True,
    )

    # =========================================================
    # 3. METRICS
    # =========================================================

    metrics = calculate_metrics(
        equity=result.equity,
        trades=result.trades,
        periods_per_year=periods_per_year,

        # BacktestResult is produced by our own trusted engine,
        # so repeating validation here is unnecessary.
        inputs_prevalidated=True,
    )

    # =========================================================
    # 4. FLATTEN RESULT
    # =========================================================

    return {
        "rsi_length": (
            parameters.rsi_length
        ),

        "oversold": (
            parameters.oversold
        ),

        "overbought": (
            parameters.overbought
        ),

        "max_bars_between_dips": (
            parameters.max_bars_between_dips
        ),

        **metrics,

        "final_equity": (
            result.final_equity
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

    Processing architecture
    -----------------------

        pandas market data
              |
              v
        validate once
              |
              v
        convert prices to NumPy
              |
              v
        build NumPy RSI cache
              |
              v
        evaluate candidates entirely with NumPy
              |
              v
        create one final pandas DataFrame

    IMPORTANT
    ---------
    This function does not select a winner.

    It returns the complete raw optimization result table so
    eligibility, robustness, ranking, and walk-forward selection
    remain independent concerns.
    """

    # =========================================================
    # 1. VALIDATE INPUTS ONCE
    # =========================================================

    _validate_market_data(
        data
    )

    if periods_per_year <= 0:
        raise ValueError(
            "periods_per_year must be greater than zero"
        )

    if backtest_config is None:
        backtest_config = BacktestConfig()

    validate_backtest_config(
        backtest_config
    )

    # =========================================================
    # 2. CONVERT MARKET DATA TO NUMPY ONCE
    # =========================================================
    #
    # Every candidate uses exactly the same Open and Close
    # arrays for this optimization window.
    #
    # Do not repeat this conversion inside the candidate loop.
    # =========================================================

    open_prices = data[
        "Open"
    ].to_numpy(
        dtype=np.float64,
        copy=False,
    )

    close_prices = data[
        "Close"
    ].to_numpy(
        dtype=np.float64,
        copy=False,
    )

    # =========================================================
    # 3. GENERATE CANDIDATES
    # =========================================================

    candidates = list(
        search_space.candidates()
    )

    if not candidates:
        raise ValueError(
            "search_space produced no valid candidates"
        )

    # =========================================================
    # 4. FIND UNIQUE RSI LENGTHS
    # =========================================================
    #
    # Example:
    #
    # RSI 14 / OS 25 / OB 70 / Gap 5
    # RSI 14 / OS 30 / OB 70 / Gap 5
    # RSI 14 / OS 35 / OB 75 / Gap 10
    #
    # All three candidates use the exact same RSI(14).
    #
    # It should therefore be calculated once.
    # =========================================================

    unique_rsi_lengths = sorted(
        {
            candidate.rsi_length
            for candidate in candidates
        }
    )

    # =========================================================
    # 5. BUILD NUMPY RSI CACHE
    # =========================================================

    close_series = data[
        "Close"
    ]

    rsi_cache: dict[
        int,
        FloatArray,
    ] = {}

    for rsi_length in unique_rsi_lengths:

        rsi_cache[
            rsi_length
        ] = _calculate_rsi(
            close=close_series,
            rsi_length=rsi_length,
        )

    # =========================================================
    # 6. EVALUATE EVERY CANDIDATE
    # =========================================================
    #
    # Everything inside this loop is now numerical.
    #
    # No:
    #
    # - DataFrames
    # - Series
    # - pandas validation
    # - pandas target construction
    # - pandas equity construction
    # - pandas trade construction
    # - pandas metric calculations
    #
    # are necessary per candidate.
    # =========================================================

    rows: list[
        dict[str, float | int]
    ] = []

    for parameters in candidates:

        rsi = rsi_cache[
            parameters.rsi_length
        ]

        row = (
            evaluate_double_dip_candidate_from_rsi(
                open_prices=open_prices,
                close_prices=close_prices,
                rsi=rsi,
                parameters=parameters,
                backtest_config=backtest_config,
                periods_per_year=periods_per_year,
            )
        )

        rows.append(
            row
        )

    # =========================================================
    # 7. CONVERT RESULTS TO PANDAS ONCE
    # =========================================================
    #
    # Candidate evaluation remains NumPy-based.
    #
    # Pandas is reintroduced only after every candidate has
    # finished so downstream research components can work with
    # a convenient tabular representation.
    # =========================================================

    results = pd.DataFrame(
        rows
    )

    # =========================================================
    # 8. RETURN RAW OPTIMIZATION RESULTS
    # =========================================================
    #
    # Do NOT:
    #
    # - sort by Sharpe
    # - select the highest-return candidate
    # - remove low-trade candidates
    # - replace infinite profit factors
    # - apply neighborhood robustness
    # - apply eligibility rules
    # - select a winner
    #
    # Those responsibilities belong to robustness.py and
    # selection.py.
    # =========================================================

    return results