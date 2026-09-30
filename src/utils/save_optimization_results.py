import pandas as pd
from pathlib import Path


def _safe_sum(result: dict | None, key: str) -> float:
    """Sum a series value from a result dict, returning 0.0 if result is None or key is missing."""
    if result is None:
        return 0.0
    val = result.get(key)
    if val is None:
        return 0.0
    if isinstance(val, pd.Series):
        return round(float(val.sum()), 2)
    return round(float(val), 2)


def _safe_scalar(result: dict | None, key: str) -> float:
    """Get a scalar value from a result dict, returning 0.0 if result is None or key is missing."""
    if result is None:
        return 0.0
    val = result.get(key)
    if val is None:
        return 0.0
    if isinstance(val, pd.Series):
        return round(float(val.iloc[-1]), 2)
    return round(float(val), 2)


def build_daily_row(day: pd.Timestamp, daily_profit: float, end_of_day_soc: float, total_profit_for_observed_period: float, result_fcr: dict | None, result_daa: dict | None, result_ida: dict | None, result_idc: dict | None) -> dict:
    """Build a single row dict with all result fields (see results/read_me.md)."""

    # --- Main ---
    row = {
        "date": day.strftime("%Y-%m-%d"),
        "daily_profit": round(daily_profit, 2),
        "end_of_day_soc": round(end_of_day_soc, 2),
        "total_profit_so_far": round(total_profit_for_observed_period, 2),
    }

    # --- FCR results ---
    row["marketed_pwr_fcr"] = _safe_sum(result_fcr, "marketed_pwr_fcr")
    row["profit_fcr_horizon"] = _safe_scalar(result_fcr, "profit_fcr_horizon")
    row["profit_fcr_commit"] = _safe_scalar(result_fcr, "profit_fcr_commit")
    row["charged_fcr"] = _safe_sum(result_fcr, "cha_fcr")
    row["discharged_fcr"] = _safe_sum(result_fcr, "discha_fcr")
    row["virtual_total_profit_fcr_commit"] = (_safe_scalar(result_fcr, "profit_fcr_commit") + _safe_scalar(result_fcr, "profit_daa_commit") + _safe_scalar(result_fcr, "profit_ida_commit") + _safe_scalar(result_fcr, "profit_idc_commit"))
    row["buy_daa_fcr"] = _safe_sum(result_fcr, "buy_daa_fcr")
    row["sell_daa_fcr"] = _safe_sum(result_fcr, "sell_daa_fcr")
    row["profit_daa_fcr_horizon"] = _safe_scalar(result_fcr, "profit_daa_horizon")
    row["profit_daa_fcr_commit"] = _safe_scalar(result_fcr, "profit_daa_commit")
    row["buy_ida_fcr"] = _safe_sum(result_fcr, "buy_ida_fcr")
    row["sell_ida_fcr"] = _safe_sum(result_fcr, "sell_ida_fcr")
    row["profit_ida_fcr_horizon"] = _safe_scalar(result_fcr, "profit_ida_horizon")
    row["profit_ida_fcr_commit"] = _safe_scalar(result_fcr, "profit_ida_commit")
    row["buy_idc_fcr"] = _safe_sum(result_fcr, "buy_idc_fcr")
    row["sell_idc_fcr"] = _safe_sum(result_fcr, "sell_idc_fcr")
    row["profit_idc_fcr_horizon"] = _safe_scalar(result_fcr, "profit_idc_horizon")
    row["profit_idc_fcr_commit"] = _safe_scalar(result_fcr, "profit_idc_commit")
    row["soc_fcr_horizon"] = _safe_scalar(result_fcr, "soc_fcr_horizon")
    row["soc_fcr_commit"] = _safe_scalar(result_fcr, "soc_fcr_commit")

    # --- DAA results ---
    row["charged_daa"] = _safe_sum(result_daa, "cha_daa")
    row["discharged_daa"] = _safe_sum(result_daa, "discha_daa")
    row["market_dispatch_daa"] = _safe_sum(result_daa, "market_dispatch_daa")
    row["buy_daa"] = _safe_sum(result_daa, "buy_daa")
    row["sell_daa"] = _safe_sum(result_daa, "sell_daa")
    row["profit_daa_horizon"] = _safe_scalar(result_daa, "profit_daa_horizon")
    row["profit_daa_commit"] = _safe_scalar(result_daa, "profit_daa_commit")
    row["buy_ida_daa"] = _safe_sum(result_daa, "buy_ida_daa")
    row["sell_ida_daa"] = _safe_sum(result_daa, "sell_ida_daa")
    row["profit_ida_daa_horizon"] = _safe_scalar(result_daa, "profit_ida_horizon")
    row["profit_ida_daa_commit"] = _safe_scalar(result_daa, "profit_ida_commit")
    row["buy_idc_daa"] = _safe_sum(result_daa, "buy_idc_daa")
    row["sell_idc_daa"] = _safe_sum(result_daa, "sell_idc_daa")
    row["profit_idc_daa_horizon"] = _safe_scalar(result_daa, "profit_idc_horizon")
    row["profit_idc_daa_commit"] = _safe_scalar(result_daa, "profit_idc_commit")
    row["soc_daa_horizon"] = _safe_scalar(result_daa, "soc_daa_horizon")
    row["soc_daa_commit"] = _safe_scalar(result_daa, "soc_daa_commit")
    row["soc_market_dispatch_daa_commit"] = _safe_scalar(result_daa, "soc_market_dispatch_daa_commit")

    # --- IDA results ---
    row["charged_ida"] = _safe_sum(result_ida, "cha_ida")
    row["discharged_ida"] = _safe_sum(result_ida, "discha_ida")
    row["market_dispatch_ida"] = _safe_sum(result_ida, "market_dispatch_ida")
    row["ida_fixed_daa_buy"] = _safe_sum(result_ida, "fixed_daa_buy")
    row["ida_fixed_daa_sell"] = _safe_sum(result_ida, "fixed_daa_sell")
    row["buy_ida"] = _safe_sum(result_ida, "buy_ida")
    row["sell_ida"] = _safe_sum(result_ida, "sell_ida")
    row["profit_ida_horizon"] = _safe_scalar(result_ida, "profit_ida_horizon")
    row["profit_ida_commit"] = _safe_scalar(result_ida, "profit_ida_commit")
    row["buy_idc_ida"] = _safe_sum(result_ida, "buy_idc_ida")
    row["sell_idc_ida"] = _safe_sum(result_ida, "sell_idc_ida")
    row["profit_idc_ida_horizon"] = _safe_scalar(result_ida, "profit_idc_horizon")
    row["profit_idc_ida_commit"] = _safe_scalar(result_ida, "profit_idc_commit")
    row["soc_ida_horizon"] = _safe_scalar(result_ida, "soc_ida_horizon")
    row["soc_ida_commit"] = _safe_scalar(result_ida, "soc_ida_commit")
    row["soc_market_dispatch_ida_commit"] = _safe_scalar(result_ida, "soc_market_dispatch_ida_commit")

    # --- IDC results ---
    row["charged_idc"] = _safe_sum(result_idc, "cha_idc")
    row["discharged_idc"] = _safe_sum(result_idc, "discha_idc")
    row["market_dispatch_idc"] = _safe_sum(result_idc, "market_dispatch_idc")
    row["idc_fixed_daa_buy"] = _safe_sum(result_idc, "fixed_daa_buy")
    row["idc_fixed_daa_sell"] = _safe_sum(result_idc, "fixed_daa_sell")
    row["idc_fixed_ida_buy"] = _safe_sum(result_idc, "fixed_ida_buy")
    row["idc_fixed_ida_sell"] = _safe_sum(result_idc, "fixed_ida_sell")
    row["buy_idc"] = _safe_sum(result_idc, "buy_idc")
    row["sell_idc"] = _safe_sum(result_idc, "sell_idc")
    row["profit_idc_horizon"] = _safe_scalar(result_idc, "profit_idc_horizon")
    row["profit_idc_commit"] = _safe_scalar(result_idc, "profit_idc_commit")
    row["soc_idc_horizon"] = _safe_scalar(result_idc, "soc_idc_horizon")
    row["soc_idc_commit"] = _safe_scalar(result_idc, "soc_idc_commit")
    row["soc_market_dispatch_idc_commit"] = _safe_scalar(result_idc, "soc_market_dispatch_idc_commit")

    return row


def init_results_csv(csv_path: Path):
    """Initialize the results CSV file. Overwrites any existing file at the same path."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if csv_path.exists():
        csv_path.unlink()


def append_daily_result(csv_path: Path, row: dict):
    """Append a single daily result row to the CSV. Writes header only if the file does not yet exist."""
    df_row = pd.DataFrame([row])
    if csv_path.exists():
        df_row.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        df_row.to_csv(csv_path, mode="w", header=True, index=False)
