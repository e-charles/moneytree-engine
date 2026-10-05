from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
PositionArray = NDArray[np.int8]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class BacktestConfig:
    """
    Configuration controlling execution assumptions.

    initial_cash:
        Starting portfolio value.

    commission_rate:
        Commission charged on each transaction as a fraction
        of transaction notional.

    slippage_rate:
        Simulated adverse price movement on execution.

        Buys execute above the quoted price.
        Sells execute below the quoted price.

    close_open_position_at_end:
        If True, any remaining long position is liquidated
        at the final bar's Close.
    """

    initial_cash: float = 100_000.0
    commission_rate: float = 0.0005
    slippage_rate: float = 0.0002
    close_open_position_at_end: bool = True


@dataclass(frozen=True)
class BacktestTrades:
    """
    Completed trades produced by the backtest.

    All fields are parallel NumPy arrays.

    entry_bars / exit_bars:
        Integer bar positions in the market-data array.

        The reporting layer may later translate these positions
        into timestamps using the original market-data index.

    exit_reasons:
        Integer encoded exit reason.

        0 = strategy target
        1 = forced end-of-data liquidation
    """

    entry_bars: IntArray
    exit_bars: IntArray

    entry_prices: FloatArray
    exit_prices: FloatArray

    units: FloatArray
    entry_notionals: FloatArray

    gross_pnl: FloatArray
    fees: FloatArray
    net_pnl: FloatArray

    return_pct: FloatArray
    bars_held: IntArray

    exit_reasons: NDArray[np.int8]

    @property
    def count(self) -> int:
        """
        Number of completed trades.
        """

        return int(
            self.entry_bars.size
        )


@dataclass(frozen=True)
class BacktestResult:
    """
    Numerical result of a backtest.

    equity:
        Portfolio value marked at each bar's Close.

    positions:
        Position held after execution at each bar's Open.

        0 = flat
        1 = long

    trades:
        Completed trade arrays.

    final_equity:
        Final portfolio value.

        Stored explicitly because it is frequently needed during
        optimization.
    """

    equity: FloatArray
    positions: PositionArray
    trades: BacktestTrades
    final_equity: float


def validate_backtest_config(
    config: BacktestConfig,
) -> None:
    """
    Validate backtest configuration.

    This should normally be called once before evaluating many
    optimization candidates.
    """

    if config.initial_cash <= 0:
        raise ValueError(
            "initial_cash must be greater than zero"
        )

    if config.commission_rate < 0:
        raise ValueError(
            "commission_rate cannot be negative"
        )

    if config.slippage_rate < 0:
        raise ValueError(
            "slippage_rate cannot be negative"
        )

    if config.slippage_rate >= 1:
        raise ValueError(
            "slippage_rate must be less than 1"
        )


def validate_backtest_arrays(
    open_prices: FloatArray,
    close_prices: FloatArray,
    target_at_close: PositionArray,
) -> None:
    """
    Validate NumPy arrays supplied to the backtesting engine.

    This function is primarily useful outside the optimizer.

    During optimization, market data and strategy contracts
    should already have been validated before entering the hot
    candidate loop.
    """

    if not isinstance(
        open_prices,
        np.ndarray,
    ):
        raise TypeError(
            "open_prices must be a NumPy ndarray"
        )

    if not isinstance(
        close_prices,
        np.ndarray,
    ):
        raise TypeError(
            "close_prices must be a NumPy ndarray"
        )

    if not isinstance(
        target_at_close,
        np.ndarray,
    ):
        raise TypeError(
            "target_at_close must be a NumPy ndarray"
        )

    if open_prices.ndim != 1:
        raise ValueError(
            "open_prices must be one-dimensional"
        )

    if close_prices.ndim != 1:
        raise ValueError(
            "close_prices must be one-dimensional"
        )

    if target_at_close.ndim != 1:
        raise ValueError(
            "target_at_close must be one-dimensional"
        )

    bar_count = open_prices.size

    if bar_count == 0:
        raise ValueError(
            "price arrays cannot be empty"
        )

    if close_prices.size != bar_count:
        raise ValueError(
            "open_prices and close_prices must "
            "have equal length"
        )

    if target_at_close.size != bar_count:
        raise ValueError(
            "target_at_close must have the same "
            "length as the price arrays"
        )

    if not np.isfinite(
        open_prices
    ).all():
        raise ValueError(
            "open_prices must contain only "
            "finite values"
        )

    if not np.isfinite(
        close_prices
    ).all():
        raise ValueError(
            "close_prices must contain only "
            "finite values"
        )

    if np.any(
        open_prices <= 0
    ):
        raise ValueError(
            "open_prices must be greater than zero"
        )

    if np.any(
        close_prices <= 0
    ):
        raise ValueError(
            "close_prices must be greater than zero"
        )

    if not np.all(
        (target_at_close == 0)
        | (target_at_close == 1)
    ):
        raise ValueError(
            "target_at_close may contain only 0 or 1"
        )


def _fill_price(
    raw_price: float,
    side: int,
    slippage_rate: float,
) -> float:
    """
    Apply adverse slippage to an execution price.

    side:
        +1 = buy
        -1 = sell
    """

    return raw_price * (
        1.0
        + side * slippage_rate
    )


def _empty_trades() -> BacktestTrades:
    """
    Construct an empty typed trade result.
    """

    return BacktestTrades(
        entry_bars=np.empty(
            0,
            dtype=np.int64,
        ),
        exit_bars=np.empty(
            0,
            dtype=np.int64,
        ),
        entry_prices=np.empty(
            0,
            dtype=np.float64,
        ),
        exit_prices=np.empty(
            0,
            dtype=np.float64,
        ),
        units=np.empty(
            0,
            dtype=np.float64,
        ),
        entry_notionals=np.empty(
            0,
            dtype=np.float64,
        ),
        gross_pnl=np.empty(
            0,
            dtype=np.float64,
        ),
        fees=np.empty(
            0,
            dtype=np.float64,
        ),
        net_pnl=np.empty(
            0,
            dtype=np.float64,
        ),
        return_pct=np.empty(
            0,
            dtype=np.float64,
        ),
        bars_held=np.empty(
            0,
            dtype=np.int64,
        ),
        exit_reasons=np.empty(
            0,
            dtype=np.int8,
        ),
    )


def run_backtest(
    open_prices: FloatArray,
    close_prices: FloatArray,
    target_at_close: PositionArray,
    config: BacktestConfig | None = None,
    *,
    inputs_prevalidated: bool = False,
) -> BacktestResult:
    """
    Run a long-only numerical backtest.

    Parameters
    ----------
    open_prices:
        One-dimensional array containing bar Open prices.

    close_prices:
        One-dimensional array containing bar Close prices.

    target_at_close:
        Close-based target positions.

        0 = flat
        1 = long

    config:
        Execution configuration.

    inputs_prevalidated:
        If False, validate the configuration and input arrays.

        The optimization engine should normally validate its
        inputs once before the candidate loop and then call this
        function with:

            inputs_prevalidated=True

    Execution timing
    ----------------
    A target generated at:

        Close[t]

    becomes executable at:

        Open[t + 1]

    Returns
    -------
    BacktestResult
        Pure NumPy backtest output.
    """

    if config is None:
        config = BacktestConfig()

    if not inputs_prevalidated:

        validate_backtest_config(
            config
        )

        validate_backtest_arrays(
            open_prices=open_prices,
            close_prices=close_prices,
            target_at_close=target_at_close,
        )

    bar_count = open_prices.size

    if bar_count == 0:
        raise ValueError(
            "backtest arrays cannot be empty"
        )

    # =========================================================
    # CLOSE TARGET -> NEXT OPEN POSITION
    # =========================================================

    position_at_open = np.empty(
        bar_count,
        dtype=np.int8,
    )

    position_at_open[0] = 0

    if bar_count > 1:
        position_at_open[1:] = (
            target_at_close[:-1]
        )

    # =========================================================
    # OUTPUT ARRAYS
    # =========================================================

    equity = np.empty(
        bar_count,
        dtype=np.float64,
    )

    positions = np.empty(
        bar_count,
        dtype=np.int8,
    )

    # =========================================================
    # PORTFOLIO STATE
    # =========================================================

    cash = float(
        config.initial_cash
    )

    units = 0.0
    current_position = 0

    # =========================================================
    # CURRENT TRADE STATE
    # =========================================================
    #
    # Sentinels are used rather than Optional values inside the
    # hot loop.
    # =========================================================

    entry_price = 0.0
    entry_fee = 0.0
    entry_notional = 0.0

    entry_bar = -1

    # =========================================================
    # COMPLETED TRADE BUFFERS
    # =========================================================
    #
    # Trades are much less frequent than bars, so Python lists
    # are reasonable temporary buffers here.
    #
    # They are converted to contiguous NumPy arrays once at the
    # end of the backtest.
    # =========================================================

    trade_entry_bars: list[int] = []
    trade_exit_bars: list[int] = []

    trade_entry_prices: list[float] = []
    trade_exit_prices: list[float] = []

    trade_units: list[float] = []
    trade_entry_notionals: list[float] = []

    trade_gross_pnl: list[float] = []
    trade_fees: list[float] = []
    trade_net_pnl: list[float] = []

    trade_returns: list[float] = []
    trade_bars_held: list[int] = []

    trade_exit_reasons: list[int] = []

    # =========================================================
    # LOCAL CONFIG VALUES
    # =========================================================
    #
    # Pull these out of the dataclass before entering the hot
    # loop so repeated attribute access is avoided.
    # =========================================================

    commission_rate = (
        config.commission_rate
    )

    slippage_rate = (
        config.slippage_rate
    )

    # =========================================================
    # EXECUTION LOOP
    # =========================================================

    for bar_number in range(
        bar_count
    ):

        desired_position = int(
            position_at_open[
                bar_number
            ]
        )

        open_price = float(
            open_prices[
                bar_number
            ]
        )

        close_price = float(
            close_prices[
                bar_number
            ]
        )

        # =====================================================
        # EXIT LONG
        # =====================================================

        if (
            current_position == 1
            and desired_position == 0
        ):

            exit_price = _fill_price(
                raw_price=open_price,
                side=-1,
                slippage_rate=slippage_rate,
            )

            exit_notional = (
                units
                * exit_price
            )

            exit_fee = (
                exit_notional
                * commission_rate
            )

            cash += (
                exit_notional
                - exit_fee
            )

            gross_pnl = (
                (
                    exit_price
                    - entry_price
                )
                * units
            )

            total_fees = (
                entry_fee
                + exit_fee
            )

            net_pnl = (
                gross_pnl
                - total_fees
            )

            if entry_notional > 0:
                return_pct = (
                    net_pnl
                    / entry_notional
                )
            else:
                return_pct = 0.0

            bars_held = (
                bar_number
                - entry_bar
            )

            # -------------------------------------------------
            # Record trade
            # -------------------------------------------------

            trade_entry_bars.append(
                entry_bar
            )

            trade_exit_bars.append(
                bar_number
            )

            trade_entry_prices.append(
                entry_price
            )

            trade_exit_prices.append(
                exit_price
            )

            trade_units.append(
                units
            )

            trade_entry_notionals.append(
                entry_notional
            )

            trade_gross_pnl.append(
                gross_pnl
            )

            trade_fees.append(
                total_fees
            )

            trade_net_pnl.append(
                net_pnl
            )

            trade_returns.append(
                return_pct
            )

            trade_bars_held.append(
                bars_held
            )

            # 0 = strategy target.
            trade_exit_reasons.append(
                0
            )

            # -------------------------------------------------
            # Reset position
            # -------------------------------------------------

            units = 0.0
            current_position = 0

            entry_price = 0.0
            entry_fee = 0.0
            entry_notional = 0.0

            entry_bar = -1

        # =====================================================
        # ENTER LONG
        # =====================================================

        if (
            current_position == 0
            and desired_position == 1
        ):

            entry_price = _fill_price(
                raw_price=open_price,
                side=1,
                slippage_rate=slippage_rate,
            )

            entry_notional = (
                cash
                / (
                    1.0
                    + commission_rate
                )
            )

            units = (
                entry_notional
                / entry_price
            )

            entry_fee = (
                entry_notional
                * commission_rate
            )

            cash -= (
                entry_notional
                + entry_fee
            )

            if abs(cash) < 1e-10:
                cash = 0.0

            current_position = 1

            entry_bar = (
                bar_number
            )

        # =====================================================
        # MARK TO MARKET
        # =====================================================

        equity[
            bar_number
        ] = (
            cash
            + units * close_price
        )

        positions[
            bar_number
        ] = current_position

    # =========================================================
    # OPTIONAL FINAL LIQUIDATION
    # =========================================================

    if (
        config.close_open_position_at_end
        and current_position == 1
    ):

        final_bar = (
            bar_count - 1
        )

        exit_price = _fill_price(
            raw_price=float(
                close_prices[
                    final_bar
                ]
            ),
            side=-1,
            slippage_rate=slippage_rate,
        )

        exit_notional = (
            units
            * exit_price
        )

        exit_fee = (
            exit_notional
            * commission_rate
        )

        cash += (
            exit_notional
            - exit_fee
        )

        gross_pnl = (
            (
                exit_price
                - entry_price
            )
            * units
        )

        total_fees = (
            entry_fee
            + exit_fee
        )

        net_pnl = (
            gross_pnl
            - total_fees
        )

        if entry_notional > 0:
            return_pct = (
                net_pnl
                / entry_notional
            )
        else:
            return_pct = 0.0

        bars_held = (
            final_bar
            - entry_bar
        )

        trade_entry_bars.append(
            entry_bar
        )

        trade_exit_bars.append(
            final_bar
        )

        trade_entry_prices.append(
            entry_price
        )

        trade_exit_prices.append(
            exit_price
        )

        trade_units.append(
            units
        )

        trade_entry_notionals.append(
            entry_notional
        )

        trade_gross_pnl.append(
            gross_pnl
        )

        trade_fees.append(
            total_fees
        )

        trade_net_pnl.append(
            net_pnl
        )

        trade_returns.append(
            return_pct
        )

        trade_bars_held.append(
            bars_held
        )

        # 1 = forced end-of-data liquidation.
        trade_exit_reasons.append(
            1
        )

        # Final portfolio is entirely cash.
        equity[
            final_bar
        ] = cash

        positions[
            final_bar
        ] = 0

    # =========================================================
    # CONSTRUCT NUMPY TRADE RESULT
    # =========================================================

    trade_count = len(
        trade_entry_bars
    )

    if trade_count == 0:

        trades = _empty_trades()

    else:

        trades = BacktestTrades(
            entry_bars=np.asarray(
                trade_entry_bars,
                dtype=np.int64,
            ),
            exit_bars=np.asarray(
                trade_exit_bars,
                dtype=np.int64,
            ),
            entry_prices=np.asarray(
                trade_entry_prices,
                dtype=np.float64,
            ),
            exit_prices=np.asarray(
                trade_exit_prices,
                dtype=np.float64,
            ),
            units=np.asarray(
                trade_units,
                dtype=np.float64,
            ),
            entry_notionals=np.asarray(
                trade_entry_notionals,
                dtype=np.float64,
            ),
            gross_pnl=np.asarray(
                trade_gross_pnl,
                dtype=np.float64,
            ),
            fees=np.asarray(
                trade_fees,
                dtype=np.float64,
            ),
            net_pnl=np.asarray(
                trade_net_pnl,
                dtype=np.float64,
            ),
            return_pct=np.asarray(
                trade_returns,
                dtype=np.float64,
            ),
            bars_held=np.asarray(
                trade_bars_held,
                dtype=np.int64,
            ),
            exit_reasons=np.asarray(
                trade_exit_reasons,
                dtype=np.int8,
            ),
        )

    return BacktestResult(
        equity=equity,
        positions=positions,
        trades=trades,
        final_equity=float(
            equity[-1]
        ),
    )