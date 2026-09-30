from __future__ import annotations
import pandas as pd
import numpy as np


#-------------------------------------------------------------
# Reindex onto a target_index with forward-fill:
# [00:00, v1; 01:00, v2; ...] -> [00:00, v1; 00:15, v1; 00:30, v1; 00:45, v1; 01:00, v2; ...]
# [00:00, v1; 00:15, v2; 00:30, v3; 00:45, v4; 01:00, v5; ...] -> [00:00, v1; 01:00, v5; ...]
# If Index is already the target-index, the index remains unchanged.
# foreward-fill -> missing values are filled with the last known value (ffill)
# Optional: Still missing values can be filled with a constant fill_value (fillna)
# Optional: divide_by can be given to scale energy values

def reindex_to_target_index(series: pd.Series, target_index: pd.DatetimeIndex, divide_by: float | None = None, fill_value: float | None = 0.0) -> pd.Series:
    
    # Index is adjusted:
    s = series.sort_index().reindex(target_index, method="ffill")

    # If fill_value is given, fill remaining NaNs with it
    if fill_value is not None:
        s = s.fillna(fill_value)
    
    # Optional scaling if Series contains energy values
    if divide_by is not None:
        s = s / float(divide_by)
        
    return s
#-------------------------------------------------------------


#-------------------------------------------------------------
# Converts an absolute profile series (e.g. MW per timestep) into (fix, nominal_value) for solph.Flow(fix=..., nominal_value=...).
# - fix must be relative in [0..1]
# - nominal_value scales it back to absolute
# Returns (None, None) if series has no positive values (nominal_value ~ 0)

def series_to_fix_and_nominal(series: pd.Series, clip_negative: bool = True, upper_clip: float = 1.0, eps: float = 1e-12,) -> tuple[np.ndarray | None, float | None]:
    
    s = series.copy().fillna(0.0)

    if clip_negative:
        s = s.clip(lower=0.0)

    nom = float(s.max())
    if nom <= eps:
        return None, None

    fix = (s / nom).clip(lower=0.0, upper=upper_clip).to_numpy(dtype=float)
    
    return fix, nom
#-------------------------------------------------------------



# ------------------------------------------------------------
# Helper functions to 
# - resolve index differences between fcr-blocks (4h), daa (1h) and ida/idc (15min)
# - transform pd.Series to solph-compatible sequences

def _expand_block_series_to_timeindex(timeindex: pd.DatetimeIndex, block_series: pd.Series) -> pd.Series:
    
    # Expand a block-based series (e.g. 6x4h FCR blocks) to match a full model timeindex (e.g. 24x1h or 96x15min).
    
    n_steps = len(timeindex)
    n_blocks = len(block_series)

    if n_steps % n_blocks != 0:
        raise ValueError(
            f"Cannot expand {n_blocks} blocks to {n_steps} timesteps."
        )

    rep = n_steps // n_blocks
    expanded = np.repeat(block_series.values, rep)

    return pd.Series(expanded, index=timeindex)

def make_storage_level_sequences(timeindex: pd.DatetimeIndex, min_soc_mwh: pd.Series, max_soc_mwh: pd.Series, capacity_mwh: float) -> tuple[list[float], list[float]]:
    
    # Convert SoC bounds in MWh per block into solph-compatible min/max_storage_level sequences (0..1).

    min_mwh = _expand_block_series_to_timeindex(timeindex, min_soc_mwh)
    max_mwh = _expand_block_series_to_timeindex(timeindex, max_soc_mwh)
    min_level = (min_mwh / capacity_mwh).clip(0.0, 1.0)
    max_level = (max_mwh / capacity_mwh).clip(0.0, 1.0)
    min_level_seq = min_level.tolist()
    max_level_seq = max_level.tolist()
    # oemof needs sequences with +1 values, if interfer_last_value=True
    min_level_seq.append(min_level_seq[-1])
    max_level_seq.append(max_level_seq[-1])

    return min_level_seq, max_level_seq

def make_flow_max_sequences(timeindex: pd.DatetimeIndex, max_power_mw: pd.Series, nominal_power_mw: float) -> list[float]:
    
    # Convert power bounds in MW per block into solph Flow.max sequences (0..1).
    
    if nominal_power_mw <= 0:
        raise ValueError("nominal_power_mw must be > 0")

    max_mw = _expand_block_series_to_timeindex(timeindex, max_power_mw)
    max_rel = (max_mw / nominal_power_mw).clip(0.0, 1.0)

    return max_rel.tolist()
# ----------------------------------------------------------