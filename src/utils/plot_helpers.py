# src/utils/plot_helpers.py

from __future__ import annotations
from typing import Optional
import numpy as np
import pandas as pd
# existing helpers:
from src.utils.transform_pd_series import reindex_to_target_index
from src.utils.transform_pd_series import _expand_block_series_to_timeindex  


def get_plot_cfg(cfg) -> dict:
    # Accepts either OmegaConf-like or dict config.
    p = getattr(cfg, "plot", None)
    if p is None and isinstance(cfg, dict):
        p = cfg.get("plot", {})
    return p or {}


def get_opt_cfg(cfg) -> dict:
    o = getattr(cfg, "optimization", None)
    if o is None and isinstance(cfg, dict):
        o = cfg.get("optimization", {})
    return o or {}


def get_market_color(cfg, market: str, fallback: str = "gray") -> str:
    p = get_plot_cfg(cfg)
    mc = p.get("market_colors", {}) or {}
    if market in mc:
        return mc[market]
    return fallback


def bar_width_ms(index: pd.DatetimeIndex) -> int:
    # Compute bar width in milliseconds from the median timestep of the index.
    if len(index) < 2:
        return int(15 * 60 * 1000)
    dt = (index[1:] - index[:-1]).median()
    return int(dt.total_seconds() * 1000)


def ensure_soc_index(soc: pd.Series, base_index: pd.DatetimeIndex) -> pd.Series:
    # If soc has one extra value (len = len(base)+1), attach it to base[-1] + dt so SoC reaches the end when infer_last_interval=True.
    soc = soc.copy()
    if len(soc) == len(base_index) + 1 and len(base_index) >= 2:
        dt = base_index[1] - base_index[0]
        end_ts = base_index[-1] + dt
        soc.index = pd.DatetimeIndex(list(base_index) + [end_ts])
    return soc


def coerce_soc_to_mwh(soc: pd.Series, capacity_mwh: float) -> pd.Series:
    # All SoC series passed to the plotter are already in MWh.
    # A value-based heuristic (e.g. soc.max() <= 1.2) is fundamentally unreliable:
    # for a 2 MWh battery at 50 % initial SoC the MWh value is 1.0, which is
    # indistinguishable from a fractional series at any simple threshold.
    # Both build_market_dispatch_soc and oemof storage_content produce absolute MWh,
    # so no conversion is needed or applied.
    return soc.copy()


def to_master_index(series: Optional[pd.Series], master_index: pd.DatetimeIndex, fill_value: float = 0.0) -> Optional[pd.Series]:
    # Reindex (ffill) and fill remaining NaNs. If series is None, return None. If series is not a pd.Series, raise TypeError.
    if series is None:
        return None
    if not isinstance(series, pd.Series):
        raise TypeError("to_master_index expects pd.Series")
    return reindex_to_target_index(series, master_index, fill_value=fill_value)


def signed_market_position_mw(buy_mw: pd.Series, sell_mw: pd.Series, master_index: pd.DatetimeIndex) -> pd.Series:
    # Buy positive, Sell negative (sell input may already be positive or negative depending on your pipeline). We enforce: pos = buy - abs(sell).
    b = to_master_index(buy_mw, master_index, fill_value=0.0)
    s = to_master_index(sell_mw, master_index, fill_value=0.0)
    if b is None or s is None:
        raise ValueError("buy_mw and sell_mw required")
    return b - s.abs()


def signed_flow_mw(charge_mw: pd.Series, discharge_mw: pd.Series, master_index: pd.DatetimeIndex) -> pd.Series:
    # Battery flow convention: charge positive, discharge negative. Enforce: flow = +charge - abs(discharge)
    c = to_master_index(charge_mw, master_index, fill_value=0.0)
    d = to_master_index(discharge_mw, master_index, fill_value=0.0)
    if c is None or d is None:
        raise ValueError("charge_mw and discharge_mw required")
    return c - d.abs()


def expand_fcr_block_series_to_master(block_series: pd.Series, master_index: pd.DatetimeIndex) -> pd.Series:
    # Expand 4h-block (or any block-based) series to master index length using existing _expand_block_series_to_timeindex helper.
    return _expand_block_series_to_timeindex(master_index, block_series)