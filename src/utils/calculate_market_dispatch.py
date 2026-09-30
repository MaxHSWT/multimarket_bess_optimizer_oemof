from __future__ import annotations
import numpy as np
import pandas as pd


def build_market_dispatch_series(buy_pwr: pd.Series, sell_pwr: pd.Series) -> pd.Series:
    """Return signed net market dispatch [MW]: positive = net charging, negative = net discharging."""
    buy = buy_pwr.fillna(0.0)
    sell = sell_pwr.fillna(0.0)
    return buy - sell


def build_market_dispatch_soc(dispatch_pwr: pd.Series, start_soc_mwh: float, dt_h: float, eff_in: float, eff_out: float) -> pd.Series:
    """Build virtual SoC [MWh] from signed dispatch power [MW], applying charge/discharge efficiency.
    Returns a series with one more element than dispatch_pwr (same index structure as oemof storage_content).
    Parameters:
    dispatch_pwr:  signed power series [MW], positive = charging, negative = discharging
    start_soc_mwh: initial SoC at the start of the horizon [MWh]
    dt_h:          time step size [h]
    eff_in:        charge (inflow) efficiency  [-]
    eff_out:       discharge (outflow) efficiency [-]
    """
    dispatch_arr = dispatch_pwr.values
    delta = np.where(
        dispatch_arr >= 0,
        dispatch_arr * eff_in * dt_h,
        dispatch_arr / eff_out * dt_h,
    )
    soc_values = np.empty(len(dispatch_arr) + 1)
    soc_values[0] = start_soc_mwh
    soc_values[1:] = start_soc_mwh + np.cumsum(delta)
    ext_index = dispatch_pwr.index.append(
        pd.DatetimeIndex([dispatch_pwr.index[-1] + pd.Timedelta(hours=dt_h)])
    )
    return pd.Series(soc_values, index=ext_index)
