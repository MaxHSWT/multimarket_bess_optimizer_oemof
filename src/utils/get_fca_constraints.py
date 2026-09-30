# src/utils/get_fca_constraints.py

# Helper functions for Flexible Connection Agreement (FCA) constraints.
# FCA limits the maximum charge/discharge power at the grid connection point.
# This module:
#   - derives FCA power limits from df_FCA as pd.Series
#   - provides a neutral (no-FCA) fallback
#   - merges FCA and FCR constraints into active power limits

from __future__ import annotations
import pandas as pd


def derive_fca_power_limits(df_FCA: pd.DataFrame) -> dict[str, pd.Series]:
    # Extract FCA power limit series from the horizon-specific df_FCA DataFrame.
    # Args: df_FCA: DataFrame with DatetimeIndex and columns 'max_cha_pwr_fca', 'max_discha_pwr_fca'.
    # Returns: dict with keys 'max_cha_pwr_fca' and 'max_discha_pwr_fca' as pd.Series.
    
    return {
        "max_cha_pwr_fca": df_FCA["max_cha_pwr_fca"].rename("max_cha_pwr_fca"),
        "max_discha_pwr_fca": df_FCA["max_discha_pwr_fca"].rename("max_discha_pwr_fca"),
    }


def neutral_fca_limits(index: pd.Index, max_charge_power_mw: float, max_discharge_power_mw: float) -> dict[str, pd.Series]:
    # Neutral FCA limits when FCA is disabled: full technical power available at all timesteps.
    return {
        "max_cha_pwr_fca": pd.Series(float(max_charge_power_mw), index=index, name="max_cha_pwr_fca"),
        "max_discha_pwr_fca": pd.Series(float(max_discharge_power_mw), index=index, name="max_discha_pwr_fca"),
    }


def compute_active_power_limits(max_cha_pwr_fca: pd.Series, max_discha_pwr_fca: pd.Series,
                                marketed_pwr_fcr: pd.Series) -> dict[str, pd.Series]:
    # Compute active (effective) power limits for energy trading from FCA and marketed FCR power.
    # FCA defines the grid connection ceiling; FCR reservation is subtracted from that ceiling.
    #   active_cha(t) = max_cha_pwr_fca(t) - marketed_pwr_fcr(B)
    #   active_discha(t) = max_discha_pwr_fca(t) - marketed_pwr_fcr(B)
    # FCA is on 15-min resolution; marketed_pwr_fcr is on 4h-block resolution.
    # Returns: dict with 'max_cha_pwr_active' and 'max_discha_pwr_active' as pd.Series (15-min).

    fca_idx = max_cha_pwr_fca.index
    # Expand marketed FCR power (4h blocks) to 15-min via forward-fill
    marketed_15 = marketed_pwr_fcr.reindex(fca_idx, method="ffill").fillna(0.0)

    # Active limit = FCA ceiling minus FCR reservation
    active_cha = (max_cha_pwr_fca - marketed_15).clip(lower=0.0)
    active_discha = (max_discha_pwr_fca - marketed_15).clip(lower=0.0)

    return {
        "max_cha_pwr_active": active_cha.rename("max_cha_pwr_active"),
        "max_discha_pwr_active": active_discha.rename("max_discha_pwr_active"),
    }
