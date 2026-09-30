# src/utils/get_fcr_constraints.py
from __future__ import annotations
import pandas as pd


def derive_fcr_constraints_from_marketed_power(marketed_pwr_fcr: pd.Series, capacity_mwh: float, max_charge_power_mw: float, max_discharge_power_mw: float) -> dict[str, pd.Series]:
    
    # Derive FCR-implied constraints (block-wise series) from marketed FCR power (MW per 4h block).
    # Uses static technical power limits as ceiling (market behaviour only, no FCA).
    # Returns dict with:
      # - min_soc_for_fcr (MWh)
      # - max_soc_for_fcr (MWh)
      # - max_cha_pwr_fcr (MW)
      # - max_discha_pwr_fcr (MW)
    
    marketed_pwr_fcr = marketed_pwr_fcr.astype(float)

    # PQ relationships
    min_soc_for_fcr = 0.25 * marketed_pwr_fcr
    max_soc_for_fcr = capacity_mwh - 0.25 * marketed_pwr_fcr

    max_cha_pwr_fcr = max_charge_power_mw - marketed_pwr_fcr
    max_discha_pwr_fcr = max_discharge_power_mw - marketed_pwr_fcr

    # Defensive clipping (numerical robustness)
    min_soc_for_fcr = min_soc_for_fcr.clip(lower=0.0, upper=capacity_mwh)
    max_soc_for_fcr = max_soc_for_fcr.clip(lower=0.0, upper=capacity_mwh)

    max_cha_pwr_fcr = max_cha_pwr_fcr.clip(lower=0.0, upper=max_charge_power_mw)
    max_discha_pwr_fcr = max_discha_pwr_fcr.clip(lower=0.0, upper=max_discharge_power_mw)

    return {
        "min_soc_for_fcr": min_soc_for_fcr.rename("min_soc_for_fcr"),
        "max_soc_for_fcr": max_soc_for_fcr.rename("max_soc_for_fcr"),
        "max_cha_pwr_fcr": max_cha_pwr_fcr.rename("max_cha_pwr_fcr"),
        "max_discha_pwr_fcr": max_discha_pwr_fcr.rename("max_discha_pwr_fcr"),
    }


def neutral_fcr_constraints(fcr_index: pd.Index, capacity_mwh: float, max_charge_power_mw: float, max_discharge_power_mw: float) -> dict[str, pd.Series]:
    
    # Neutral constraints if no FCR is marketed.
    
    return {
        "min_soc_for_fcr": pd.Series(0.0, index=fcr_index, name="min_soc_for_fcr"),
        "max_soc_for_fcr": pd.Series(float(capacity_mwh), index=fcr_index, name="max_soc_for_fcr"),
        "max_cha_pwr_fcr": pd.Series(float(max_charge_power_mw), index=fcr_index, name="max_cha_pwr_fcr"),
        "max_discha_pwr_fcr": pd.Series(float(max_discharge_power_mw), index=fcr_index, name="max_discha_pwr_fcr"),
    }
