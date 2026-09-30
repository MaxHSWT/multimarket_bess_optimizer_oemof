from __future__ import annotations
import logging
import sys
from pathlib import Path
from typing import Tuple
import pandas as pd
import hydra


log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level cache: stores imported DataFrames per market (RAM only)
# ---------------------------------------------------------------------------
_import_cache: dict[str, pd.DataFrame] = {}     # initialized as a private variable, defined as empty dictionary holding pd series by a string key


# ---------------------------------------------------------------------------
# Import markets layer — reads CSV files once, parses and caches full time-series
# ---------------------------------------------------------------------------

def _get_data_dir(cfg) -> Path:
    # Resolve data directory relative to Hydra original working directory: base_dir / subfolder
    base_dir = Path(str(cfg.data.base_dir))
    if not base_dir.is_absolute():
        try:
            base_dir = Path(hydra.utils.get_original_cwd()) / base_dir
        except ValueError:
            # Hydra not fully initialized (e.g., debug mode) — use current working directory
            base_dir = Path.cwd() / base_dir
    subfolder = str(getattr(cfg.data, "subfolder", ""))
    if subfolder:
        base_dir = base_dir / subfolder
    return base_dir


def _import_energy_market(cfg, market: str) -> pd.DataFrame:
    # Import DAA / IDA / IDC from a single CSV file with a timestamp column (mainly 15 min resolution).
    # Returns full-range DataFrame with DatetimeIndex (in cfg.simulation.timezone) and standardized 'price' column.
    tz = str(getattr(cfg.simulation, "timezone", "Europe/Berlin"))
    mcfg = cfg.data.markets[market]
    path = _get_data_dir(cfg) / str(mcfg.filename)
    sep = str(getattr(mcfg, "separator", ";"))
    ts_col = str(mcfg.timestamp_column)
    price_col = str(mcfg.price_column)

    if not path.exists():
        raise FileNotFoundError(f"Missing data file for {market}: {path}")

    log.info("Importing %s from %s (col=%s)", market, path, price_col)

    df = pd.read_csv(path, sep=sep, decimal=",")        # main reading command for data from csv file
    if df.empty:
        raise ValueError(f"Loaded empty file: {path}")

    # Validate columns
    if ts_col not in df.columns:
        raise KeyError(f"Expected timestamp column '{ts_col}' in {path}, got {list(df.columns)}")
    if price_col not in df.columns:
        raise KeyError(f"Expected price column '{price_col}' in {path}, got {list(df.columns)}")

    # Build DatetimeIndex from timestamp column
    idx = pd.DatetimeIndex(pd.to_datetime(df[ts_col], utc=True, errors="coerce"))   # parse timestamps as UTC first (to avoid DST (Daylight Saving Time) issues)
    # Attention: Only works if timestemps are in ISO format!
    if idx.isna().any():
        raise ValueError(f"Found unparsable timestamps in column '{ts_col}' of {path}")
    idx = idx.tz_convert(tz)        # convert to timezone

    # Select and rename price column
    df = df[[price_col]].copy()
    df.index = idx
    df.index.name = None
    df = df.rename(columns={price_col: "price"})
    df["price"] = pd.to_numeric(df["price"], errors="coerce")    # convert prices to numbers

    # Sort for robustness (no resampling here — done in load layer to avoid cross-DST alignment issues)
    df = df.sort_index()
    df["price"] = df["price"].astype(float)

    log.info("%s imported: %d rows, tz=%s, price-range=%.2f..%.2f",
             market, len(df), df.index.tz, float(df["price"].min()), float(df["price"].max()))

    return df[["price"]]


def _import_capacity_market(cfg) -> pd.DataFrame:
    # Import FCR from a capacity-market CSV with date + interval columns (mainly 4 h resolution, tailored to the data export of regelleistung.net).
    # Parses date_column (e.g. "02.10.2025") + interval_column (e.g. "NEGPOS_00_04") into timestamps.
    # Returns full-range DataFrame with DatetimeIndex (in cfg.simulation.timezone) and standardized 'price' column.
    tz = str(getattr(cfg.simulation, "timezone", "Europe/Berlin"))
    mcfg = cfg.data.markets["FCR"]
    path = _get_data_dir(cfg) / str(mcfg.filename)
    sep = str(getattr(mcfg, "separator", ";"))
    date_col = str(mcfg.date_column)
    interval_col = str(mcfg.interval_column)
    price_col = str(mcfg.price_column)

    if not path.exists():
        raise FileNotFoundError(f"Missing data file for FCR: {path}")

    log.info("Importing FCR from %s (date=%s, interval=%s, price=%s)", path, date_col, interval_col, price_col)

    df = pd.read_csv(path, sep=sep, decimal=",")
    if df.empty:
        raise ValueError(f"Loaded empty file: {path}")

    # Validate columns
    for col in (date_col, interval_col, price_col):
        if col not in df.columns:
            raise KeyError(f"Expected column '{col}' in {path}, got {list(df.columns)}")

    # Parse timestamps from date + interval: "02.10.2025" + "NEGPOS_00_04" -> 2025-10-02 00:00:00+tz
    dates = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce")
    if dates.isna().any():
        raise ValueError(f"Found unparsable dates in column '{date_col}' of {path}")

    # Extract start hour from interval_column: "NEGPOS_00_04" -> 0, "NEGPOS_12_16" -> 12
    start_hours = df[interval_col].str.split("_").str[-2].astype(int)
    timestamps = dates + pd.to_timedelta(start_hours, unit="h")
    timestamps = timestamps.dt.tz_localize(tz)

    # Build DataFrame with price column
    result = df[[price_col]].copy()
    result.index = timestamps
    result = result.rename(columns={price_col: "price"})
    # Explicitly normalise comma-decimal strings (e.g. "63,10") before numeric conversion,
    # because pandas' decimal="," in read_csv does not always apply to object-typed columns.
    result["price"] = pd.to_numeric(
        result["price"].astype(str).str.replace(",", ".", regex=False),
        errors="coerce",
    )

    # Sort for robustness (no resampling here — done in load layer to avoid cross-DST alignment issues)
    result = result.sort_index()
    result["price"] = result["price"].astype(float)

    log.info("FCR imported: %d rows, tz=%s, price-range=%.2f..%.2f",
             len(result), result.index.tz, float(result["price"].min()), float(result["price"].max()))

    return result[["price"]]

def _import_market(cfg, market: str) -> pd.DataFrame:
    # Dispatcher: imports market data from CSV and caches the result.
    # Returns cached DataFrame on subsequent calls for the same market.
    if market in _import_cache:
        log.debug("Cache hit for %s", market)
        return _import_cache[market]

    if market == "FCR":
        df = _import_capacity_market(cfg)
    else:
        df = _import_energy_market(cfg, market)

    _import_cache[market] = df
    return df


# ---------------------------------------------------------------------------
# Import FCA layer — reads FCA csv once, validates and caches the profile matrix
# ---------------------------------------------------------------------------

# Month abbreviation mapping (German month abbreviations used in FCA CSV column headers)
_FCA_MONTH_MAP = {
    1: "jan", 2: "feb", 3: "mae", 4: "apr", 5: "mai", 6: "jun",
    7: "jul", 8: "aug", 9: "sep", 10: "okt", 11: "nov", 12: "dez",
}

def _get_fca_dir(cfg) -> Path:
    # Resolve FCA data directory relative to Hydra original working directory.
    gc = cfg.grid_constraints
    base_dir = Path(str(gc.base_dir))
    if not base_dir.is_absolute():
        try:
            base_dir = Path(hydra.utils.get_original_cwd()) / base_dir
        except ValueError:
            base_dir = Path.cwd() / base_dir
    subfolder = str(getattr(gc, "subfolder", ""))
    if subfolder:
        base_dir = base_dir / subfolder
    return base_dir

def _import_fca(cfg) -> pd.DataFrame:
    # Import and validate the FCA CSV file. Returns a DataFrame with index 0..95 and all monthly columns.
    # Caches the result in _import_cache under key 'FCA'.

    if "FCA" in _import_cache:
        log.debug("Cache hit for FCA")
        return _import_cache["FCA"]

    gc = cfg.grid_constraints
    path = _get_fca_dir(cfg) / str(gc.filename)
    sep = str(getattr(gc, "separator", "\t"))
    if not path.exists():
        raise FileNotFoundError(f"Missing FCA data file: {path}")

    log.info("Importing FCA from %s", path)

    df = pd.read_csv(path, sep=sep, decimal=",")
    if df.empty:
        raise ValueError(f"Loaded empty FCA file: {path}")

    # --- Validation ---
    # 1. Check 'Index' column exists (case-insensitive lookup)
    index_col = None
    for col in df.columns:
        if col.strip().lower() == "index":
            index_col = col
            break
    if index_col is None:
        raise KeyError(f"FCA file must contain an 'Index' column, got {list(df.columns)}")
    df[index_col] = pd.to_numeric(df[index_col], errors="coerce").astype(int)
    expected_index = list(range(96))
    if list(df[index_col]) != expected_index:
        raise ValueError(f"FCA 'Index' column must contain exactly 0..95, got {list(df[index_col][:5])}...")

    # 2. Check all monthly columns exist
    required_cols = []
    for m in range(1, 13):
        abbr = _FCA_MONTH_MAP[m]
        required_cols.append(f"cha_{abbr}")
        required_cols.append(f"discha_{abbr}")
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"FCA file is missing required columns: {missing}. Available: {list(df.columns)}")

    # 3. Set index, keep only data columns
    df = df.set_index(index_col)
    df = df[required_cols]

    # 4. Ensure numeric and check value ranges
    for col in required_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if df.isna().any().any():
        nan_cols = df.columns[df.isna().any()].tolist()
        raise ValueError(f"FCA file contains NaN values in columns: {nan_cols}")
    if (df < -1e-9).any().any() or (df > 1.0 + 1e-9).any().any():
        raise ValueError("FCA factors must be in [0, 1].")

    # 5. Check exact row count
    if len(df) != 96:
        raise ValueError(f"FCA file must have exactly 96 rows, got {len(df)}")
    log.info("FCA imported successfully: %d rows, %d columns", len(df), len(df.columns))

    _import_cache["FCA"] = df   # cache the validated FCA DataFrame under key 'FCA'

    return df


# ---------------------------------------------------------------------------
# Load markets layer — slices cached DataFrames to rolling-horizon windows
# ---------------------------------------------------------------------------

def load_market_horizon(cfg, market: str, horizon_start: pd.Timestamp, lookahead_hours: int) -> pd.DataFrame:
    # Slice the cached full time-series for a market to [horizon_start, horizon_start + lookahead_hours).
    # Resamples to market frequency after slicing (avoids cross-DST alignment issues).
    # Returns a DataFrame indexed in cfg.simulation.timezone with column 'price'.
    tz = str(getattr(cfg.simulation, "timezone", "Europe/Berlin"))
    horizon_start = pd.Timestamp(horizon_start)
    if horizon_start.tzinfo is None:
        horizon_start = horizon_start.tz_localize(tz)
    else:
        horizon_start = horizon_start.tz_convert(tz)

    lookahead_hours = int(lookahead_hours)
    if lookahead_hours <= 0:
        raise ValueError("lookahead_hours must be > 0")

    horizon_end = horizon_start + pd.Timedelta(hours=lookahead_hours)
    df = _import_market(cfg, market)    # collects data from cache or imports if not cached yet

    # Slice to exact rolling horizon: [start, end); pandas slicing is inclusive on both ends, so we do end-epsilon.
    end_eps = horizon_end - pd.Timedelta(microseconds=1)
    df = df.loc[horizon_start:end_eps]

    # Resample to market frequency (on the small window — safe w.r.t. DST)
    freq = str(cfg.data.markets[market].frequency)

    # Resolution check: configured frequency must not be finer than the raw data resolution.
    if len(df) >= 2:
        raw_step_s = pd.Series(df.index).diff().dropna().dt.total_seconds().median()
        req_step_s = pd.tseries.frequencies.to_offset(freq).nanos / 1e9
        if raw_step_s > req_step_s + 1:  # raw step is coarser than requested (1 s tolerance)
            raise ValueError(
                f"{market}: configured frequency '{freq}' ({req_step_s/60:.0f} min) is finer than the "
                f"data resolution ({raw_step_s/60:.0f} min). "
                f"Provide higher-resolution data or set a coarser frequency in the config."
            )

    df = df.resample(freq).mean()       # if frequency higher than original, this will introduce NaNs; if lower, it will aggregate with average value (e.g. 15 min -> 1 h)
    df["price"] = df["price"].astype(float).ffill().bfill()

    # Final sanity
    if df.empty:
        raise ValueError(f"Window for {market} is empty after slicing: start={horizon_start}, end={horizon_end}")

    return df[["price"]]



def load_all_markets_horizon(cfg, horizon_start: pd.Timestamp, lookahead_hours: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    
    # Convenience loader returning (df_FCR, df_DAA, df_IDA, df_IDC) for one rolling horizon.
    
    df_FCR = load_market_horizon(cfg, "FCR", horizon_start, lookahead_hours)
    df_DAA = load_market_horizon(cfg, "DAA", horizon_start, lookahead_hours)
    df_IDA = load_market_horizon(cfg, "IDA", horizon_start, lookahead_hours)
    df_IDC = load_market_horizon(cfg, "IDC", horizon_start, lookahead_hours)
    
    return df_FCR, df_DAA, df_IDA, df_IDC


# ---------------------------------------------------------------------------
# Load FCA layer — slices cached DataFrame to rolling-horizon window
# ---------------------------------------------------------------------------

def load_fca_horizon(cfg, horizon_start: pd.Timestamp, lookahead_hours: int, max_charge_power: float, max_discharge_power: float) -> pd.DataFrame:
    """Build a horizon-specific df_FCA with absolute power limits from the cached FCA profile matrix.
    Returns a DataFrame with DatetimeIndex (15-min freq) over the full horizon and columns:
        - max_cha_pwr_fca  (MW)
        - max_discha_pwr_fca  (MW)
        - cha_factor  (0..1)
        - discha_factor  (0..1)
    Correctly handles month boundaries within the horizon (e.g. Jan 31 -> Feb 1 in lookahead)."""

    tz = str(getattr(cfg.simulation, "timezone", "Europe/Berlin"))
    horizon_start = pd.Timestamp(horizon_start)
    if horizon_start.tzinfo is None:
        horizon_start = horizon_start.tz_localize(tz)
    else:
        horizon_start = horizon_start.tz_convert(tz)

    horizon_end = horizon_start + pd.Timedelta(hours=lookahead_hours)

    # Build 15-min master index for the full horizon
    master_index = pd.date_range(start=horizon_start, end=horizon_end, freq="15min", tz=tz, inclusive="left")

    # Import cached FCA profile matrix (index 0..95, columns cha_jan, discha_jan, ...)
    fca_raw = _import_fca(cfg)

    # Build factor series per timestep: lookup the correct month column and 15-min slot
    cha_factors = pd.Series(index=master_index, dtype=float)
    discha_factors = pd.Series(index=master_index, dtype=float)

    for ts in master_index:
        month_num = ts.month
        abbr = _FCA_MONTH_MAP[month_num]
        # 15-min slot within the day: hours * 4 + quarter
        slot = ts.hour * 4 + ts.minute // 15
        cha_factors.at[ts] = float(fca_raw.at[slot, f"cha_{abbr}"])
        discha_factors.at[ts] = float(fca_raw.at[slot, f"discha_{abbr}"])

    # Build absolute power limits
    df_FCA = pd.DataFrame({
        "cha_factor": cha_factors,
        "discha_factor": discha_factors,
        "max_cha_pwr_fca": cha_factors * float(max_charge_power),
        "max_discha_pwr_fca": discha_factors * float(max_discharge_power),
    }, index=master_index)

    log.info("FCA horizon slice: %d timesteps, range=[%s .. %s]", len(df_FCA), df_FCA.index.min(), df_FCA.index.max())

    return df_FCA


def load_all_markets_and_fca_horizon(cfg, 
                                     horizon_start: pd.Timestamp, 
                                     lookahead_hours: int,
                                     max_charge_power: float, 
                                     max_discharge_power: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Convenience loader returning (df_FCR, df_DAA, df_IDA, df_IDC, df_FCA) for one rolling horizon.
    # If FCA is disabled in config, returns a neutral df_FCA with full technical power limits.

    df_FCR, df_DAA, df_IDA, df_IDC = load_all_markets_horizon(cfg, horizon_start, lookahead_hours)

    apply_fca = bool(getattr(cfg.grid_constraints, "apply_fca", False)) if hasattr(cfg, "grid_constraints") else False

    if apply_fca:
        df_FCA = load_fca_horizon(cfg, horizon_start, lookahead_hours, max_charge_power, max_discharge_power)
    else:
        # Neutral FCA: full power available at all timesteps
        tz = str(getattr(cfg.simulation, "timezone", "Europe/Berlin"))
        hs = pd.Timestamp(horizon_start)
        if hs.tzinfo is None:
            hs = hs.tz_localize(tz)
        horizon_end = hs + pd.Timedelta(hours=lookahead_hours)
        master_index = pd.date_range(start=hs, end=horizon_end, freq="15min", tz=tz, inclusive="left")
        df_FCA = pd.DataFrame({
            "cha_factor": 1.0,
            "discha_factor": 1.0,
            "max_cha_pwr_fca": float(max_charge_power),
            "max_discha_pwr_fca": float(max_discharge_power),
        }, index=master_index)

    return df_FCR, df_DAA, df_IDA, df_IDC, df_FCA


# ---------------------------------------------------------------------------
# Data availability checks
# ---------------------------------------------------------------------------

def check_data_availability(cfg, start_day: pd.Timestamp, end_day: pd.Timestamp, lookahead_hours: int, markets=("FCR", "DAA", "IDA", "IDC")) -> None:
    
    # Checks whether all required CSV files exist and cover the full simulation horizon.
    # Loads data into cache as a side-effect (avoids double-read later).
    # Raises FileNotFoundError / ValueError with a clear message if data is missing.
    
    tz = str(getattr(cfg.simulation, "timezone", "Europe/Berlin"))
    start_day = pd.Timestamp(start_day)
    end_day = pd.Timestamp(end_day)

    if start_day.tzinfo is None:
        start_day = start_day.tz_localize(tz)
    else:
        start_day = start_day.tz_convert(tz)

    if end_day.tzinfo is None:
        end_day = end_day.tz_localize(tz)
    else:
        end_day = end_day.tz_convert(tz)

    start_day = start_day.normalize()
    end_day = end_day.normalize()

    lookahead_hours = int(lookahead_hours)
    if lookahead_hours <= 0:
        raise ValueError("lookahead_hours must be > 0")

    # Need data up to end_day + lookahead (rolling horizon window can extend beyond end_day)
    horizon_end = end_day + pd.Timedelta(hours=lookahead_hours)

    problems = []
    for m in markets:
        # 1. Check file exists
        mcfg = cfg.data.markets[m]
        path = _get_data_dir(cfg) / str(mcfg.filename)
        if not path.exists():
            problems.append(f"- {m}: file not found: {path}")
            continue

        # 2. Import and check temporal coverage (also populates cache)
        df = _import_market(cfg, m)
        data_start = df.index.min()
        data_end = df.index.max()

        # Last required index timestamp: the data point at (horizon_end - freq) covers [t, t+freq) = [..., horizon_end)
        freq = str(cfg.data.markets[m].frequency)
        last_needed_idx = horizon_end - pd.Timedelta(freq)

        if data_start > start_day:
            problems.append(f"- {m}: data starts at {data_start}, but simulation needs {start_day}")
        if data_end < last_needed_idx:
            problems.append(f"- {m}: data ends at {data_end}, but simulation needs up to {last_needed_idx} (last index at {freq} resolution)")

    if problems:
        msg = (
            f"Data availability check failed for simulation "
            f"{start_day.date()}..{end_day.date()} with lookahead={lookahead_hours}h:\n"
            + "\n".join(problems)
        )
        raise FileNotFoundError(msg)

def check_fca_availability(cfg) -> None:
    # Check whether the FCA CSV file exists and is valid. Loads into cache as side-effect.
    apply_fca = bool(getattr(cfg.grid_constraints, "apply_fca", False)) if hasattr(cfg, "grid_constraints") else False
    if not apply_fca:
        log.info("FCA is disabled in config — skipping FCA availability check.")
        return
    _import_fca(cfg)
    log.info("FCA data availability check passed.")


# ---------------------------------------------------------------------------
# Debug test function — callable directly from terminal
# ---------------------------------------------------------------------------

def _debug_test_day(date_str: str) -> None:
    # Quick test: import all markets and print the sliced DataFrames for a given day.
    from hydra import initialize, compose
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    with initialize(version_base=None, config_path="../../configs"):
        cfg = compose(config_name="config")

    tz = str(cfg.simulation.timezone)
    lookahead_hours = int(cfg.optimization.rolling_horizon.lookahead_hours)
    horizon_start = pd.Timestamp(date_str).tz_localize(tz).normalize()

    print(f"\n{'='*80}")
    print(f"  Debug test for {horizon_start.date()} | lookahead={lookahead_hours}h | tz={tz}")
    print(f"{'='*80}")

    df_FCR, df_DAA, df_IDA, df_IDC = load_all_markets_horizon(cfg, horizon_start, lookahead_hours)

    for name, df in [("FCR", df_FCR), ("DAA", df_DAA), ("IDA", df_IDA), ("IDC", df_IDC)]:
        print(f"\n--- {name} ---  shape={df.shape}  "
              f"range=[{df.index.min()} .. {df.index.max()}]")
        print(df.to_string())

    print(f"\n{'='*80}")
    print("  Debug test completed successfully.")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.utils.import_and_load_data <date>")
        print("Example: python -m src.utils.import_and_load_data 2025-10-02")
        sys.exit(1)
    logging.basicConfig(level=logging.INFO)
    _debug_test_day(sys.argv[1])

