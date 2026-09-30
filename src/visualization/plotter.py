# src/visualization/plotter.py
from __future__ import annotations
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from src.utils.plot_helpers import (
    get_plot_cfg,
    get_opt_cfg,
    get_market_color,
    bar_width_ms,
    ensure_soc_index,
    coerce_soc_to_mwh,
    to_master_index,
    signed_market_position_mw,
    signed_flow_mw,
    expand_fcr_block_series_to_master,
)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class MarketPlotData:
    """Container for all data related to one market (FCR, DAA, IDA, IDC) needed for plotting."""
    price_eur_mwh: Optional[pd.Series] = None
    buy_mw: Optional[pd.Series] = None
    sell_mw: Optional[pd.Series] = None
    charge_mw: Optional[pd.Series] = None
    discharge_mw: Optional[pd.Series] = None
    dispatch_mw: Optional[pd.Series] = None   # signed dispatch (+ charge, − discharge); energy markets only
    soc_mwh: Optional[pd.Series] = None


@dataclass
class FCRLimits:
    """FCR-specific limits (power and SoC) plotted as dashed lines."""
    max_charge_mw: Optional[pd.Series] = None
    max_discharge_mw: Optional[pd.Series] = None
    min_soc_mwh: Optional[pd.Series] = None
    max_soc_mwh: Optional[pd.Series] = None


@dataclass
class FCALimits:
    """FCA-specific power limits plotted as long-dashed step lines."""
    max_charge_mw: Optional[pd.Series] = None
    max_discharge_mw: Optional[pd.Series] = None


@dataclass
class ActivePowerLimits:
    """Active (effective) power limits for energy trading after FCR reservation."""
    max_charge_mw: Optional[pd.Series] = None
    max_discharge_mw: Optional[pd.Series] = None


@dataclass
class HorizonPlotRecord:
    """All data needed to plot one horizon/day."""
    date_label: str
    master_index: pd.DatetimeIndex
    markets: Dict[str, MarketPlotData]
    fcr_limits: Optional[FCRLimits] = None
    fca_limits: Optional[FCALimits] = None
    active_limits: Optional[ActivePowerLimits] = None


# ---------------------------------------------------------------------------
# Key mapping: which result-dict keys correspond to buy/sell/charge/discharge/soc per market
# ---------------------------------------------------------------------------

_MARKET_KEYS = {
    "FCR": {
        "price_col": "price_fcr",          # column in df_all_markets
        "buy_mw": "marketed_pwr_fcr",     # FCR marketed power shown as positive position
        "sell_mw": None,
        "charge_mw": "cha_fcr_pwr",
        "discharge_mw": "discha_fcr_pwr",
        "soc_mwh": "soc_fcr_horizon",
    },
    "DAA": {
        "price_col": "price_daa",
        "buy_mw": "buy_daa_pwr",
        "sell_mw": "sell_daa_pwr",
        "charge_mw": None,
        "discharge_mw": None,
        "dispatch_mw": "market_dispatch_daa_pwr",
        "soc_mwh": "soc_market_dispatch_daa",
    },
    "IDA": {
        "price_col": "price_ida",
        "buy_mw": "buy_ida_pwr",
        "sell_mw": "sell_ida_pwr",
        "charge_mw": None,
        "discharge_mw": None,
        "dispatch_mw": "market_dispatch_ida_pwr",
        "soc_mwh": "soc_market_dispatch_ida",
    },
    "IDC": {
        "price_col": "price_idc",
        "buy_mw": "buy_idc_pwr",
        "sell_mw": "sell_idc_pwr",
        "charge_mw": None,
        "discharge_mw": None,
        "dispatch_mw": "market_dispatch_idc_pwr",
        "soc_mwh": "soc_market_dispatch_idc",
    },
}

# ---------------------------------------------------------------------------
# Profit series key mapping: which result-dict keys contain the per-period
# profit series for each market (horizon = full lookahead, commit = day window)
# ---------------------------------------------------------------------------

_PROFIT_KEYS = {
    "FCR": {
        "horizon_series": "profit_fcr_horizon_series",
        "commit_series":  "profit_fcr_commit_series",
    },
    "DAA": {
        "horizon_series": "profit_daa_horizon_series",
        "commit_series":  "profit_daa_commit_series",
    },
    "IDA": {
        "horizon_series": "profit_ida_horizon_series",
        "commit_series":  "profit_ida_commit_series",
    },
    "IDC": {
        "horizon_series": "profit_idc_horizon_series",
        "commit_series":  "profit_idc_commit_series",
    },
}


class MultiMarketPlotter:
    """
    Central plot class for the multimarket battery optimization.

    Public API:
      - plot_day(day, result_fcr, result_daa, result_ida, result_idc)
      - plot_horizon(day, result_fcr, result_daa, result_ida, result_idc)
      - plot_profit_index(csv_path=None)

    Internally builds HorizonPlotRecord from the raw result dicts and
    the price DataFrame stored during __init__.
    """

    def __init__(self, cfg, df_all_markets: pd.DataFrame, capacity_mwh: float, nominal_power_mw: float):
        self.log = logging.getLogger(__name__)
        self.cfg = cfg
        self.df_all_markets = df_all_markets    # 15-min master index with price_fcr, price_daa, price_ida, price_idc
        self.capacity_mwh = float(capacity_mwh)
        self.nominal_power_mw = float(nominal_power_mw)

        self.pcfg = get_plot_cfg(cfg)
        self.ocfg = get_opt_cfg(cfg)

        # Resolve nested optimization flags: cfg.optimization.optimize_<M>.run_<M>
        self.run_flags = {}
        for m in ["FCR", "DAA", "IDA", "IDC"]:
            sub = self.ocfg.get(f"optimize_{m}", {})
            if hasattr(sub, "__getattr__"):  # OmegaConf DictConfig
                self.run_flags[m] = bool(getattr(sub, f"run_{m}", False))
            elif isinstance(sub, dict):
                self.run_flags[m] = bool(sub.get(f"run_{m}", False))
            else:
                self.run_flags[m] = False

        # Commit hours for slicing day out of horizon
        rh = self.ocfg.get("rolling_horizon", {})
        if hasattr(rh, "__getattr__"):
            self.commit_hours = int(getattr(rh, "commit_hours", 24))
        else:
            self.commit_hours = int(rh.get("commit_hours", 24))

        # Plotting config section
        plot_cfg = self._get_plotting_cfg()
        self.output_dir = Path(plot_cfg.get("output_dir", "results/plots"))

        # Style settings
        op = (self.pcfg.get("opacity") or {})
        self.opacity_bars = float(op.get("bars", 0.55))
        self.opacity_price = float(op.get("price", 0.9))
        self.opacity_soc = float(op.get("soc", 0.9))
        self.opacity_limits = float(op.get("fcr_limits", 0.7))

        lw = (self.pcfg.get("line_width") or {})
        self.lw_price = int(lw.get("price", 2))
        self.lw_soc = int(lw.get("soc", 2))
        self.lw_limits = int(lw.get("limits", 2))

        self.font_family = self.pcfg.get("font_family", "Arial Black")
        self.bg_color = self.pcfg.get("bg_color", "white")
        self.grid_color = self.pcfg.get("grid_color", "lightgray")
        self.grid_color_soc = self.pcfg.get("grid_color_soc", "gray")
        self.vertical_lines_hours = int(self.pcfg.get("vertical_lines_hours", 3))
        self.adapt_soc_axis_scale = bool(plot_cfg.get("adapt_soc_axis_scale", False))

        # Export settings (for write_html toImageButtonOptions)
        exp = (plot_cfg.get("export") or {})
        self.export_format = str(exp.get("format", "jpeg"))
        self.export_scale = float(exp.get("scale", 3))
        self.export_width = int(exp.get("width", 1600))
        self.export_height = int(exp.get("height", 850))

    # ------------------------------------------------------------------
    # Config helpers
    # ------------------------------------------------------------------

    def _get_plotting_cfg(self) -> dict:
        """Read the 'plotting' section from cfg (supports OmegaConf and plain dict)."""
        p = getattr(self.cfg, "plotting", None)
        if p is None and isinstance(self.cfg, dict):
            p = self.cfg.get("plotting", {})
        if p is None:
            return {}
        # OmegaConf -> convert to dict for .get() calls
        if hasattr(p, "keys"):
            return {k: (p[k] if not hasattr(p[k], "keys") else p[k]) for k in p}
        return {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def plot_day(self, day: pd.Timestamp, result_fcr: dict | None, result_daa: dict | None, result_ida: dict | None, result_idc: dict | None) -> str | None:
        """Plot only the committed window (e.g. 24h) for a given day."""
        record = self._build_record_from_results(day, result_fcr, result_daa, result_ida, result_idc)
        if record is None:
            return None
        record = self._slice_to_day(record)
        results_by_market = {"FCR": result_fcr, "DAA": result_daa, "IDA": result_ida, "IDC": result_idc}
        profit_series = self._build_profit_series(results_by_market, record.master_index, "commit_series") or None
        title = f"Day Plot – {record.date_label}"
        fig = self._build_multimarket_figure(record, title, profit_series=profit_series)
        out_path = self.output_dir / f"plot_{record.date_label}_day.html"
        return self._save_figure(fig, out_path)

    def plot_horizon(self, day: pd.Timestamp, result_fcr: dict | None, result_daa: dict | None, result_ida: dict | None, result_idc: dict | None) -> str | None:
        """Plot the full lookahead horizon for a given day."""
        record = self._build_record_from_results(day, result_fcr, result_daa, result_ida, result_idc)
        if record is None:
            return None
        results_by_market = {"FCR": result_fcr, "DAA": result_daa, "IDA": result_ida, "IDC": result_idc}
        profit_series = self._build_profit_series(results_by_market, record.master_index, "horizon_series") or None
        title = f"Horizon Plot – {record.date_label}"
        fig = self._build_multimarket_figure(record, title, profit_series=profit_series)
        out_path = self.output_dir / f"plot_{record.date_label}_horizon.html"
        return self._save_figure(fig, out_path)

    def plot_profit_index(self, csv_path: str | Path | None = None) -> str | None:
        """Plot cumulative profit index from the results CSV."""
        if csv_path is None:
            csv_path = Path("results") / "optimization_results.csv"
        csv_path = Path(csv_path)
        df = self._load_results_csv(csv_path)
        if df is None:
            return None
        fig = self._build_profit_index_figure(df)
        out_path = self.output_dir / "plot_profit-index.html"
        return self._save_figure(fig, out_path)

    def plot_daily_profit_development(self, day: pd.Timestamp, result_fcr: dict | None, result_daa: dict | None, result_ida: dict | None, result_idc: dict | None) -> tuple[str | None, str | None]:
        """Plot the cumulative profit development for each active market during a daily optimization.

        Produces two HTML files:
          - horizon plot  (plot_horizon_{date}___profit-development.html)
          - daily plot    (plot_day_{date}___profit-development.html)

        The horizon plot uses the full lookahead profit series; the daily plot uses the commit-window series only.  Both show cumulative sums so the
        traces represent the running profit total over the respective window.
        """
        results_by_market = {"FCR": result_fcr, "DAA": result_daa, "IDA": result_ida, "IDC": result_idc}
        date_label = str(day.date())

        # Build target indexes for reindexing after cumsum.
        # Reindexing + ffill extends coarse series (e.g. FCR 4h blocks) to the
        # full window so the plotted line continues to the end of the day.
        master_idx = self.df_all_markets.index
        commit_end = master_idx[0] + pd.Timedelta(hours=self.commit_hours)
        commit_idx = master_idx[master_idx < commit_end]

        horizon_series: Dict[str, pd.Series] = {}
        commit_series: Dict[str, pd.Series] = {}

        for m in ["FCR", "DAA", "IDA", "IDC"]:
            if not self.run_flags.get(m, False):
                continue
            res = results_by_market[m]
            if res is None:
                continue
            keys = _PROFIT_KEYS[m]
            s_h = self._safe_series(res, keys["horizon_series"])
            s_c = self._safe_series(res, keys["commit_series"])
            if s_h is not None and len(s_h) > 0:
                horizon_series[m] = s_h.cumsum().reindex(master_idx).ffill()
            if s_c is not None and len(s_c) > 0:
                commit_series[m] = s_c.cumsum().reindex(commit_idx).ffill()

        path_horizon = None
        if horizon_series:
            fig = self._build_profit_development_figure(
                horizon_series,
                f"Cumulative Profit Development (Horizon) \u2013 {date_label}",
            )
            out_path = self.output_dir / f"plot_{date_label}_horizon-profit-development.html"
            path_horizon = self._save_figure(fig, out_path)

        path_day = None
        if commit_series:
            fig = self._build_profit_development_figure(
                commit_series,
                f"Cumulative Profit Development (Day) \u2013 {date_label}",
            )
            out_path = self.output_dir / f"plot_{date_label}_day-profit-development.html"
            path_day = self._save_figure(fig, out_path)

        return path_horizon, path_day

    # ------------------------------------------------------------------
    # Record building from result dicts
    # ------------------------------------------------------------------

    def _build_record_from_results(self, day: pd.Timestamp, result_fcr: dict | None, result_daa: dict | None, result_ida: dict | None, result_idc: dict | None) -> HorizonPlotRecord | None:
        """Build a HorizonPlotRecord directly from the raw result dicts + price DataFrame."""
        master_index = self.df_all_markets.index
        results_by_market = {"FCR": result_fcr, "DAA": result_daa, "IDA": result_ida, "IDC": result_idc}

        markets: Dict[str, MarketPlotData] = {}
        for m in ["FCR", "DAA", "IDA", "IDC"]:
            if not self.run_flags.get(m, False):
                continue
            res = results_by_market[m]
            if res is None:
                continue

            keys = _MARKET_KEYS[m]

            # Price from df_all_markets
            price_col = keys["price_col"]
            price = self.df_all_markets[price_col] if price_col in self.df_all_markets.columns else None

            # Buy/Sell (MW)
            buy = self._safe_series(res, keys["buy_mw"])
            sell = self._safe_series(res, keys["sell_mw"])

            # Charge/Discharge (MW) — used by FCR; energy markets use dispatch_mw
            charge = self._safe_series(res, keys["charge_mw"])
            discharge = self._safe_series(res, keys["discharge_mw"])

            # Signed dispatch series (energy markets only; positive = charge, negative = discharge)
            dispatch = self._safe_series(res, keys.get("dispatch_mw"))

            # SoC (MWh)
            soc = self._safe_series(res, keys["soc_mwh"])

            markets[m] = MarketPlotData(
                price_eur_mwh=price,
                buy_mw=buy,
                sell_mw=sell,
                charge_mw=charge,
                discharge_mw=discharge,
                dispatch_mw=dispatch,
                soc_mwh=soc,
            )

        # FCR limits
        fcr_limits = None
        if self.run_flags.get("FCR", False) and result_fcr is not None:
            fcr_limits = self._build_fcr_limits(result_fcr, master_index)

        # FCA limits (check all result dicts, FCA series may come from any stage)
        # Only show FCA traces when FCA is actually enabled in config
        fca_limits = None
        active_limits = None
        apply_fca = bool(getattr(getattr(self.cfg, "grid_constraints", None), "apply_fca", False))
        if apply_fca:
            for res in (result_fcr, result_daa, result_ida, result_idc):
                if res is not None:
                    if fca_limits is None:
                        fca_limits = self._build_fca_limits(res, master_index)
                    if active_limits is None:
                        active_limits = self._build_active_limits(res, master_index)
                    if fca_limits is not None and active_limits is not None:
                        break

        if not markets:
            self.log.warning("No active markets with data for %s – skipping plot.", day)
            return None

        return HorizonPlotRecord(
            date_label=str(day.date()),
            master_index=master_index,
            markets=markets,
            fcr_limits=fcr_limits,
            fca_limits=fca_limits,
            active_limits=active_limits,
        )

    def _build_fcr_limits(self, result_fcr: dict, master_index: pd.DatetimeIndex) -> FCRLimits:
        """Extract FCR limits from result_fcr and expand 4h blocks to master index."""
        max_cha = self._safe_series(result_fcr, "max_cha_pwr_fcr")
        max_discha = self._safe_series(result_fcr, "max_discha_pwr_fcr")
        min_soc = self._safe_series(result_fcr, "min_soc_for_fcr")
        max_soc = self._safe_series(result_fcr, "max_soc_for_fcr")

        # Expand from 4h-block index to 15-min master index
        if max_cha is not None:
            max_cha = expand_fcr_block_series_to_master(max_cha, master_index)
        if max_discha is not None:
            max_discha = expand_fcr_block_series_to_master(max_discha, master_index)
        if min_soc is not None:
            min_soc = expand_fcr_block_series_to_master(min_soc, master_index)
        if max_soc is not None:
            max_soc = expand_fcr_block_series_to_master(max_soc, master_index)

        return FCRLimits(
            max_charge_mw=max_cha,
            max_discharge_mw=max_discha,
            min_soc_mwh=min_soc,
            max_soc_mwh=max_soc,
        )

    def _build_fca_limits(self, result: dict, master_index: pd.DatetimeIndex) -> FCALimits | None:
        """Extract FCA power limits from any result dict. Already on 15-min index, no expansion needed."""
        max_cha = self._safe_series(result, "max_cha_pwr_fca")
        max_discha = self._safe_series(result, "max_discha_pwr_fca")
        if max_cha is None and max_discha is None:
            return None
        return FCALimits(max_charge_mw=max_cha, max_discharge_mw=max_discha)

    def _build_active_limits(self, result: dict, master_index: pd.DatetimeIndex) -> ActivePowerLimits | None:
        """Extract active power limits from any energy-market result dict."""
        max_cha = self._safe_series(result, "max_cha_pwr_active")
        max_discha = self._safe_series(result, "max_discha_pwr_active")
        if max_cha is None and max_discha is None:
            return None
        return ActivePowerLimits(max_charge_mw=max_cha, max_discharge_mw=max_discha)

    @staticmethod
    def _safe_series(result: dict | None, key: str | None) -> pd.Series | None:
        """Safely extract a pd.Series from a result dict. Returns None if key is None, missing, or value is not a Series."""
        if result is None or key is None:
            return None
        val = result.get(key)
        if val is None:
            return None
        if isinstance(val, pd.Series):
            return val
        return None

    # ------------------------------------------------------------------
    # Slicing: horizon -> day (commit window only)
    # ------------------------------------------------------------------

    def _slice_to_day(self, record: HorizonPlotRecord) -> HorizonPlotRecord:
        """Slice a full-horizon record down to the committed window (commit_hours)."""
        idx = record.master_index
        commit_end = idx[0] + pd.Timedelta(hours=self.commit_hours)

        day_idx = idx[idx < commit_end]
        if len(day_idx) == 0:
            return record

        markets_sliced: Dict[str, MarketPlotData] = {}
        for m, mpd in record.markets.items():
            markets_sliced[m] = MarketPlotData(
                price_eur_mwh=self._slice_series(mpd.price_eur_mwh, day_idx),
                buy_mw=self._slice_series(mpd.buy_mw, day_idx),
                sell_mw=self._slice_series(mpd.sell_mw, day_idx),
                charge_mw=self._slice_series(mpd.charge_mw, day_idx),
                discharge_mw=self._slice_series(mpd.discharge_mw, day_idx),
                dispatch_mw=self._slice_series(mpd.dispatch_mw, day_idx),
                soc_mwh=self._slice_series(mpd.soc_mwh, day_idx),
            )

        fcr_sliced = None
        if record.fcr_limits is not None:
            lim = record.fcr_limits
            fcr_sliced = FCRLimits(
                max_charge_mw=self._slice_series(lim.max_charge_mw, day_idx),
                max_discharge_mw=self._slice_series(lim.max_discharge_mw, day_idx),
                min_soc_mwh=self._slice_series(lim.min_soc_mwh, day_idx),
                max_soc_mwh=self._slice_series(lim.max_soc_mwh, day_idx),
            )

        fca_sliced = None
        if record.fca_limits is not None:
            fca = record.fca_limits
            fca_sliced = FCALimits(
                max_charge_mw=self._slice_series(fca.max_charge_mw, day_idx),
                max_discharge_mw=self._slice_series(fca.max_discharge_mw, day_idx),
            )

        active_sliced = None
        if record.active_limits is not None:
            al = record.active_limits
            active_sliced = ActivePowerLimits(
                max_charge_mw=self._slice_series(al.max_charge_mw, day_idx),
                max_discharge_mw=self._slice_series(al.max_discharge_mw, day_idx),
            )

        return HorizonPlotRecord(
            date_label=record.date_label,
            master_index=day_idx,
            markets=markets_sliced,
            fcr_limits=fcr_sliced,
            fca_limits=fca_sliced,
            active_limits=active_sliced,
        )

    @staticmethod
    def _slice_series(s: pd.Series | None, day_idx: pd.DatetimeIndex) -> pd.Series | None:
        """Slice a series to the given day index range, keeping tolerance for SoC (+1 element)."""
        if s is None:
            return None
        start, end = day_idx[0], day_idx[-1]
        # Allow one extra timestep for SoC series (len = len(idx)+1)
        dt = day_idx[1] - day_idx[0] if len(day_idx) > 1 else pd.Timedelta(minutes=15)
        return s.loc[(s.index >= start) & (s.index <= end + dt)]

    # ------------------------------------------------------------------
    # Figure building (shared by plot_day and plot_horizon)
    # ------------------------------------------------------------------

    def _build_profit_series(self, results_by_market: Dict[str, dict | None], target_idx: pd.DatetimeIndex, series_key: str) -> Dict[str, pd.Series]:
        """Compute cumulative profit series per active market, reindexed to target_idx."""
        series: Dict[str, pd.Series] = {}
        for m in ["FCR", "DAA", "IDA", "IDC"]:
            if not self.run_flags.get(m, False):
                continue
            res = results_by_market.get(m)
            if res is None:
                continue
            s = self._safe_series(res, _PROFIT_KEYS[m][series_key])
            if s is not None and len(s) > 0:
                series[m] = s.cumsum().reindex(target_idx).ffill()
        return series

    def _build_multimarket_figure(self, record: HorizonPlotRecord, title: str, profit_series: Dict[str, pd.Series] | None = None) -> go.Figure:
        idx = record.master_index
        w_ms = bar_width_ms(idx)

        n_rows = 3 if profit_series else 2
        specs = [[{"secondary_y": True}]] * n_rows
        if profit_series:
            specs[2] = [{"secondary_y": False}]

        fig = make_subplots(
            rows=n_rows, cols=1, shared_xaxes=True, vertical_spacing=0.06,
            specs=specs,
            row_heights=[1.0, 1.0, 0.5] if n_rows == 3 else [1.0, 1.0],
        )

        active_markets = [m for m in ["FCR", "DAA", "IDA", "IDC"]
                          if self.run_flags.get(m, False) and m in record.markets]

        # --- TOP row 1: Price traces first (legend order: Price → Position → Results) ---
        for m in active_markets:
            mpd = record.markets[m]
            color = get_market_color(self.cfg, m)
            if mpd.price_eur_mwh is not None:
                p = to_master_index(mpd.price_eur_mwh, idx, fill_value=None)
                if p is not None:
                    fig.add_trace(
                        go.Scatter(
                            x=p.index, y=p.values,
                            name=f"Prices {m} [€/MWh]", mode="lines",
                            line=dict(color=color, width=self.lw_price, shape="hv"),
                            opacity=self.opacity_price,
                            hovertemplate="<b>%{x}</b><br>Price: %{y:.2f} €/MWh<extra></extra>",
                        ),
                        row=1, col=1, secondary_y=True,
                    )

        # --- TOP row 2: Market Position bars ---
        for m in active_markets:
            mpd = record.markets[m]
            color = get_market_color(self.cfg, m)
            if mpd.buy_mw is not None or mpd.sell_mw is not None:
                buy_s = mpd.buy_mw if mpd.buy_mw is not None else pd.Series(0.0, index=idx)
                sell_s = mpd.sell_mw if mpd.sell_mw is not None else pd.Series(0.0, index=idx)
                pos = signed_market_position_mw(buy_s, sell_s, idx)
                fig.add_trace(
                    go.Bar(
                        x=pos.index, y=pos.values,
                        name=f"Market-Position {m} [MW]",
                        marker_color=color, opacity=self.opacity_bars, width=w_ms, offset=0,
                        hovertemplate="<b>%{x}</b><br>Market Position: %{y:.2f} MW<extra></extra>",
                    ),
                    row=1, col=1, secondary_y=False,
                )

        # --- BOTTOM results: FCR limits first, then energy market Battery Flow + SoC ---
        if ("FCR" in active_markets) and (record.fcr_limits is not None):
            self._add_fcr_limit_traces(fig, record.fcr_limits, idx)

        for m in active_markets:
            if m == "FCR":
                continue
            mpd = record.markets[m]
            color = get_market_color(self.cfg, m)

            if mpd.dispatch_mw is not None:
                flow = to_master_index(mpd.dispatch_mw, idx, fill_value=0.0)
                fig.add_trace(
                    go.Bar(
                        x=flow.index, y=flow.values,
                        name=f"Battery-Flow {m} [MW]",
                        marker_color=color, opacity=self.opacity_bars, width=w_ms, offset=0,
                        hovertemplate="<b>%{x}</b><br>Battery Flow: %{y:.2f} MW<extra></extra>",
                    ),
                    row=2, col=1, secondary_y=False,
                )
            elif mpd.charge_mw is not None and mpd.discharge_mw is not None:
                flow = signed_flow_mw(mpd.charge_mw, mpd.discharge_mw, idx)
                fig.add_trace(
                    go.Bar(
                        x=flow.index, y=flow.values,
                        name=f"Battery-Flow {m} [MW]",
                        marker_color=color, opacity=self.opacity_bars, width=w_ms, offset=0,
                        hovertemplate="<b>%{x}</b><br>Battery Flow: %{y:.2f} MW<extra></extra>",
                    ),
                    row=2, col=1, secondary_y=False,
                )

            if mpd.soc_mwh is not None:
                soc = mpd.soc_mwh.copy()
                soc = coerce_soc_to_mwh(soc, self.capacity_mwh)
                soc = ensure_soc_index(soc, idx)
                soc_plot = soc
                if len(soc_plot) == len(idx):
                    soc_plot = to_master_index(soc_plot, idx, fill_value=None)

                fig.add_trace(
                    go.Scatter(
                        x=soc_plot.index, y=soc_plot.values,
                        name=f"SoC {m} [MWh]", mode="lines",
                        line=dict(color=color, width=self.lw_soc, shape="hv"),
                        opacity=self.opacity_soc,
                        hovertemplate="<b>%{x}</b><br>SoC: %{y:.2f} MWh<extra></extra>",
                    ),
                    row=2, col=1, secondary_y=True,
                )

        # FCA grid connection limits (shown regardless of which markets are active)
        if record.fca_limits is not None:
            self._add_fca_limit_traces(fig, record.fca_limits, idx)

        # Active power limits (FCA minus FCR reservation)
        if record.active_limits is not None:
            self._add_active_limit_traces(fig, record.active_limits, idx)

        # --- Profit Development (row 3 if available) ---
        if profit_series:
            self._add_profit_traces(fig, profit_series, row=3)

        self._apply_layout(fig, record, active_markets, title, profit_series=profit_series)
        return fig

    def _add_fcr_limit_traces(self, fig: go.Figure, lim: FCRLimits, idx: pd.DatetimeIndex) -> None:
        c = get_market_color(self.cfg, "FCR")

        if lim.max_charge_mw is not None:
            s = to_master_index(lim.max_charge_mw, idx, fill_value=None)
            fig.add_trace(
                go.Scatter(
                    x=s.index, y=s.values,
                    name="max. charge FCR [MW]", mode="lines",
                    line=dict(color=c, width=self.lw_limits, shape="hv", dash="dash"),
                    opacity=self.opacity_limits,
                    hovertemplate="<b>%{x}</b><br>max charge: %{y:.2f} MW<extra></extra>",
                ),
                row=2, col=1, secondary_y=False,
            )
        if lim.max_discharge_mw is not None:
            s = to_master_index(lim.max_discharge_mw, idx, fill_value=None)
            fig.add_trace(
                go.Scatter(
                    x=s.index, y=(-s.abs()).values,
                    name="max. discharge FCR [MW]", mode="lines",
                    line=dict(color=c, width=self.lw_limits, shape="hv", dash="dash"),
                    opacity=self.opacity_limits,
                    hovertemplate="<b>%{x}</b><br>max discharge: %{y:.2f} MW<extra></extra>",
                ),
                row=2, col=1, secondary_y=False,
            )
        if lim.min_soc_mwh is not None:
            s = to_master_index(lim.min_soc_mwh, idx, fill_value=None)
            fig.add_trace(
                go.Scatter(
                    x=s.index, y=s.values,
                    name="min. SoC FCR [MWh]", mode="lines",
                    line=dict(color=c, width=self.lw_limits, shape="hv", dash="dot"),
                    opacity=self.opacity_limits,
                    hovertemplate="<b>%{x}</b><br>min SoC: %{y:.2f} MWh<extra></extra>",
                ),
                row=2, col=1, secondary_y=True,
            )
        if lim.max_soc_mwh is not None:
            s = to_master_index(lim.max_soc_mwh, idx, fill_value=None)
            fig.add_trace(
                go.Scatter(
                    x=s.index, y=s.values,
                    name="max. SoC FCR [MWh]", mode="lines",
                    line=dict(color=c, width=self.lw_limits, shape="hv", dash="dot"),
                    opacity=self.opacity_limits,
                    hovertemplate="<b>%{x}</b><br>max SoC: %{y:.2f} MWh<extra></extra>",
                ),
                row=2, col=1, secondary_y=True,
            )

    def _add_fca_limit_traces(self, fig: go.Figure, lim: FCALimits, idx: pd.DatetimeIndex) -> None:
        """Add FCA grid connection limit traces as long-dashed step lines."""
        c = "#888888"  # neutral grey for grid limits

        if lim.max_charge_mw is not None:
            s = to_master_index(lim.max_charge_mw, idx, fill_value=None)
            fig.add_trace(
                go.Scatter(
                    x=s.index, y=s.values,
                    name="max. charge FCA [MW]", mode="lines",
                    line=dict(color=c, width=self.lw_limits, shape="hv", dash="longdash"),
                    opacity=self.opacity_limits,
                    hovertemplate="<b>%{x}</b><br>FCA max charge: %{y:.2f} MW<extra></extra>",
                ),
                row=2, col=1, secondary_y=False,
            )
        if lim.max_discharge_mw is not None:
            s = to_master_index(lim.max_discharge_mw, idx, fill_value=None)
            fig.add_trace(
                go.Scatter(
                    x=s.index, y=(-s.abs()).values,
                    name="max. discharge FCA [MW]", mode="lines",
                    line=dict(color=c, width=self.lw_limits, shape="hv", dash="longdash"),
                    opacity=self.opacity_limits,
                    hovertemplate="<b>%{x}</b><br>FCA max discharge: %{y:.2f} MW<extra></extra>",
                ),
                row=2, col=1, secondary_y=False,
            )

    def _add_active_limit_traces(self, fig: go.Figure, lim: ActivePowerLimits, idx: pd.DatetimeIndex) -> None:
        """Add active (effective) power limit traces as dashed dark-grey step lines."""
        c = "#444444"  # dark grey

        if lim.max_charge_mw is not None:
            s = to_master_index(lim.max_charge_mw, idx, fill_value=None)
            fig.add_trace(
                go.Scatter(
                    x=s.index, y=s.values,
                    name="max. charge eff. [MW]", mode="lines",
                    line=dict(color=c, width=self.lw_limits, shape="hv", dash="dash"),
                    opacity=self.opacity_limits,
                    hovertemplate="<b>%{x}</b><br>max charge eff.: %{y:.2f} MW<extra></extra>",
                ),
                row=2, col=1, secondary_y=False,
            )
        if lim.max_discharge_mw is not None:
            s = to_master_index(lim.max_discharge_mw, idx, fill_value=None)
            fig.add_trace(
                go.Scatter(
                    x=s.index, y=(-s.abs()).values,
                    name="max. discharge eff. [MW]", mode="lines",
                    line=dict(color=c, width=self.lw_limits, shape="hv", dash="dash"),
                    opacity=self.opacity_limits,
                    hovertemplate="<b>%{x}</b><br>max discharge eff.: %{y:.2f} MW<extra></extra>",
                ),
                row=2, col=1, secondary_y=False,
            )

    def _add_profit_traces(self, fig: go.Figure, profit_series: Dict[str, pd.Series], row: int) -> None:
        """Add per-market cumulative profit lines and a total line to the given subplot row."""
        total_series: pd.Series | None = None
        for m in ["FCR", "DAA", "IDA", "IDC"]:
            if m not in profit_series:
                continue
            s = profit_series[m]
            color = get_market_color(self.cfg, m)
            fig.add_trace(
                go.Scatter(
                    x=s.index, y=s.values,
                    name=f"Profit {m} [\u20ac]", mode="lines",
                    line=dict(color=color, width=self.lw_price, shape="hv"),
                    opacity=self.opacity_price,
                    hovertemplate=f"<b>%{{x}}</b><br>{m} Profit: %{{y:.2f}} \u20ac<extra></extra>",
                ),
                row=row, col=1,
            )
            total_series = s.copy() if total_series is None else total_series.add(s, fill_value=0.0)

        if total_series is not None:
            fig.add_trace(
                go.Scatter(
                    x=total_series.index, y=total_series.values,
                    name="Total Profit [\u20ac]", mode="lines",
                    line=dict(color="#333333", width=self.lw_price + 1, shape="hv", dash="dot"),
                    opacity=self.opacity_price,
                    hovertemplate="<b>%{x}</b><br>Total Profit: %{y:.2f} \u20ac<extra></extra>",
                ),
                row=row, col=1,
            )

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _apply_layout(self, fig: go.Figure, record: HorizonPlotRecord, active_markets: list[str], title: str, profit_series: Dict[str, pd.Series] | None = None) -> None:
        idx = record.master_index

        pmax = max(1.0, self.nominal_power_mw)
        y_mw = 1.1 * pmax

        price_max = 1.0
        for m in active_markets:
            s = record.markets[m].price_eur_mwh
            if s is not None and len(s):
                price_max = max(price_max, float(pd.Series(s).abs().max()))
        y_price = 1.1 * price_max

        if self.adapt_soc_axis_scale:
            soc_vals: list[float] = []
            for m in active_markets:
                if m == "FCR":
                    continue
                s = record.markets[m].soc_mwh
                if s is not None and len(s):
                    finite = s.dropna()
                    if len(finite):
                        soc_vals.append(float(finite.max()))
            if record.fcr_limits is not None:
                for lim_s in [record.fcr_limits.min_soc_mwh, record.fcr_limits.max_soc_mwh]:
                    if lim_s is not None and len(lim_s):
                        finite = lim_s.dropna()
                        if len(finite):
                            soc_vals.append(float(finite.max()))
            max_soc = max(soc_vals) if soc_vals else self.capacity_mwh
            y_soc = 1.1 * max(1.0, max_soc)
        else:
            y_soc = 1.1 * max(1.0, self.capacity_mwh)

        n_rows = 3 if profit_series else 2
        fig.update_layout(
            height=1050 if n_rows == 3 else 675,
            width=900,
            bargap=0.0,
            barmode="overlay",
            font=dict(family=self.font_family),
            plot_bgcolor=self.bg_color,
            paper_bgcolor=self.bg_color,
            legend=dict(
                orientation="h",
                yanchor="bottom", y=1.02,
                xanchor="left", x=0,
                font=dict(size=10),
                entrywidthmode="fraction",
                entrywidth=0.25,
            ),
            margin=dict(l=60, r=60, t=40, b=40),
        )

        fig.add_annotation(
            text=title,
            xref="paper", yref="paper",
            x=1.04, y=0.5,
            showarrow=False,
            textangle=-90,
            xanchor="center",
            yanchor="middle",
            font=dict(family=self.font_family, size=13),
        )

        fig.update_yaxes(title_text="Market Position (MW)", row=1, col=1, secondary_y=False, range=[-y_mw, y_mw], gridcolor=self.grid_color, zeroline=True, zerolinecolor="black", zerolinewidth=1)
        fig.update_yaxes(title_text="Price (€/MWh)", row=1, col=1, secondary_y=True, range=[-y_price, y_price], showgrid=False, zeroline=True, zerolinecolor="black", zerolinewidth=1)
        fig.update_yaxes(title_text="Battery Flow (MW)", row=2, col=1, secondary_y=False, range=[-y_mw, y_mw], gridcolor=self.grid_color, zeroline=True, zerolinecolor="black", zerolinewidth=1)
        fig.update_yaxes(title_text="SoC (MWh)", row=2, col=1, secondary_y=True, range=[-y_soc, y_soc], showgrid=False, zeroline=True, zerolinecolor="black", zerolinewidth=1)
        if profit_series:
            # Compute total series (sum across all markets) to include in range calculation
            _total: pd.Series | None = None
            for _s in profit_series.values():
                _total = _s.copy() if _total is None else _total.add(_s, fill_value=0.0)
            all_vals = [float(v) for s in profit_series.values() for v in s.values if pd.notna(v)]
            if _total is not None:
                all_vals += [float(v) for v in _total.values if pd.notna(v)]
            if all_vals:
                y_p_min, y_p_max = min(all_vals), max(all_vals)
                y_profit_range = [1.1 * min(y_p_min, -1.0), 1.1 * max(y_p_max, 1.0)]
            else:
                y_profit_range = [-100, 100]
            fig.update_yaxes(title_text="Profit (\u20ac)", row=3, col=1, range=y_profit_range, gridcolor=self.grid_color, zeroline=True, zerolinecolor="black", zerolinewidth=1)
            fig.update_xaxes(title_text="Time", row=3, col=1, showgrid=True, gridcolor=self.grid_color, showline=True, linecolor="black", ticks="outside", tickcolor="black")
            fig.update_xaxes(showticklabels=True, showgrid=True, gridcolor=self.grid_color, showline=True, linecolor="black", ticks="outside", tickcolor="black", row=2, col=1)
            fig.update_xaxes(showticklabels=True, showgrid=True, gridcolor=self.grid_color, showline=True, linecolor="black", ticks="outside", tickcolor="black", row=1, col=1)
        else:
            fig.update_xaxes(title_text="Time", row=2, col=1, showgrid=True, gridcolor=self.grid_color, showline=True, linecolor="black", ticks="outside", tickcolor="black")
            fig.update_xaxes(showticklabels=True, showgrid=True, gridcolor=self.grid_color, showline=True, linecolor="black", ticks="outside", tickcolor="black", row=1, col=1)

        if self.vertical_lines_hours and len(idx) > 1:
            start, end = idx[0], idx[-1]
            step = pd.Timedelta(hours=self.vertical_lines_hours)
            t = start
            while t <= end:
                fig.add_vline(x=t, line_width=1, line_dash="dot", line_color=self.grid_color, opacity=0.5)
                t += step

    # ------------------------------------------------------------------
    # Profit Index Plot
    # ------------------------------------------------------------------

    def _load_results_csv(self, csv_path: Path) -> pd.DataFrame | None:
        if not csv_path.exists():
            self.log.warning("Results CSV not found: %s", csv_path)
            return None
        df = pd.read_csv(csv_path)
        if df.empty or "date" not in df.columns:
            self.log.warning("Results CSV is empty or missing 'date' column.")
            return None
        return df

    def _build_profit_index_figure(self, df: pd.DataFrame) -> go.Figure:
        fig = go.Figure()

        # Main line: daily_profit with markers
        hover_cols = ["daily_profit", "profit_fcr_commit", "profit_daa_commit",
                      "profit_ida_commit", "profit_idc_commit",
                      "total_profit_so_far", "end_of_day_soc"]
        hover_parts = []
        for col in hover_cols:
            if col in df.columns:
                hover_parts.append(f"{col}: %{{customdata[{len(hover_parts)}]:.2f}}")

        customdata_cols = [col for col in hover_cols if col in df.columns]
        customdata = df[customdata_cols].values if customdata_cols else None

        hovertemplate = "<b>%{x}</b><br>" + "<br>".join(hover_parts) + "<extra></extra>"

        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df["daily_profit"],
                mode="lines+markers",
                name="Daily Profit [€]",
                line=dict(color="steelblue", width=2),
                marker=dict(size=7),
                customdata=customdata,
                hovertemplate=hovertemplate,
            )
        )

        # Cumulative profit as secondary trace
        if "total_profit_so_far" in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df["date"],
                    y=df["total_profit_so_far"],
                    mode="lines",
                    name="Total Profit [€]",
                    line=dict(color="darkblue", width=2),
                    yaxis="y2",
                    hovertemplate="<b>%{x}</b><br>Total Profit: %{y:.2f} €<extra></extra>",
                )
            )

        # Overall average daily profit (horizontal dashed lightgrey line)
        avg_profit = df["daily_profit"].mean()
        fig.add_trace(
            go.Scatter(
                x=[df["date"].iloc[0], df["date"].iloc[-1]],
                y=[avg_profit, avg_profit],
                mode="lines",
                name=f"Avg. Daily Profit [{avg_profit:.2f} €]",
                line=dict(color="lightgrey", width=1.5, dash="dash"),
                hovertemplate="<b>Avg. Daily Profit</b>: %{y:.2f} €<extra></extra>",
            )
        )

        # Monthly average daily profit (green small-dashed step line)
        dates = pd.to_datetime(df["date"])
        df_tmp = df.copy()
        df_tmp["_date"] = dates
        monthly_avg = df_tmp.groupby(df_tmp["_date"].dt.to_period("M"))["daily_profit"].mean()

        if len(monthly_avg) > 0:
            step_x: list = []
            step_y: list = []
            for period, avg_val in monthly_avg.items():
                month_start = period.start_time.strftime("%Y-%m-%d")
                month_end = period.end_time.strftime("%Y-%m-%d")
                step_x.append(month_start)
                step_y.append(avg_val)
                step_x.append(month_end)
                step_y.append(avg_val)

            fig.add_trace(
                go.Scatter(
                    x=step_x,
                    y=step_y,
                    mode="lines",
                    name="Monthly Avg. Profit [€]",
                    line=dict(color="green", width=1.5, dash="dashdot"),
                    hovertemplate="<b>%{x}</b><br>Monthly Avg.: %{y:.2f} €<extra></extra>",
                )
            )

        fig.update_layout(
            height=500,
            font=dict(family=self.font_family),
            plot_bgcolor=self.bg_color,
            paper_bgcolor=self.bg_color,
            xaxis=dict(title="Date", showgrid=True, gridcolor=self.grid_color, showline=True, linecolor="black", ticks="outside", tickcolor="black"),
            yaxis=dict(title="Daily Profit (€)", showgrid=True, gridcolor=self.grid_color, zeroline=True, zerolinecolor="black", zerolinewidth=1),
            yaxis2=dict(title="Cumulative Profit (€)", overlaying="y", side="right", showgrid=False),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            margin=dict(l=60, r=120, t=40, b=40),
        )

        fig.add_annotation(
            text="Profit Index",
            xref="paper", yref="paper",
            x=1.10, y=0.5,
            showarrow=False,
            textangle=-90,
            xanchor="center",
            yanchor="middle",
            font=dict(family=self.font_family, size=14),
        )

        return fig

    # ------------------------------------------------------------------
    # Profit Development Figure
    # ------------------------------------------------------------------

    def _build_profit_development_figure(self, series_by_market: Dict[str, pd.Series], title: str) -> go.Figure:
        """Build a per-period profit development figure for a set of markets.

        One scatter line per market (market color), shared y-axis spanning the
        global min/max of all series, and horizontal help-lines at round profit
        marks.  Hover shows market name, timestamp and profit value.
        """
        fig = go.Figure()

        # --- Global y-range across all series (including total) ---
        all_values: list[float] = []
        total_series: pd.Series | None = None
        for s in series_by_market.values():
            all_values.extend(float(v) for v in s.values if pd.notna(v))
            if total_series is None:
                total_series = s.copy()
            else:
                total_series = total_series.add(s, fill_value=0.0)

        if total_series is not None:
            all_values.extend(float(v) for v in total_series.values if pd.notna(v))

        if not all_values:
            return fig

        y_min = min(all_values)
        y_max = max(all_values)
        y_pad = max(abs(y_max - y_min) * 0.05, 1.0)
        y_range = [y_min - y_pad, y_max + y_pad]

        # --- One line trace per market ---
        for m in ["FCR", "DAA", "IDA", "IDC"]:   # preserve canonical draw order
            if m not in series_by_market:
                continue
            s = series_by_market[m]
            color = get_market_color(self.cfg, m)
            fig.add_trace(
                go.Scatter(
                    x=s.index,
                    y=s.values,
                    name=f"{m} Profit [\u20ac]",
                    mode="lines",
                    line=dict(color=color, width=2, shape="hv"),
                    opacity=self.opacity_price,
                    hovertemplate="<b>%{x}</b><br>" + m + " Profit: %{y:.2f} \u20ac<extra></extra>",
                )
            )

        # --- Total profit trace ---
        if total_series is not None:
            fig.add_trace(
                go.Scatter(
                    x=total_series.index,
                    y=total_series.values,
                    name="Total Profit [\u20ac]",
                    mode="lines",
                    line=dict(color="black", width=2.5, shape="hv"),
                    opacity=self.opacity_price,
                    hovertemplate="<b>%{x}</b><br>Total Profit: %{y:.2f} \u20ac<extra></extra>",
                )
            )

        # --- Horizontal help-lines at relevant profit marks ---
        step = self._profit_helpline_step(y_min, y_max)
        first_mark = math.floor(y_min / step) * step
        t = first_mark
        while t <= y_max + step * 0.5:
            fig.add_hline(
                y=t,
                line_width=1,
                line_dash="dot",
                line_color=self.grid_color,
                opacity=0.6,
                annotation_text=f"{t:.0f} \u20ac",
                annotation_position="right",
                annotation_font=dict(family=self.font_family, size=11),
            )
            t = round(t + step, 10)

        # --- Layout ---
        fig.update_layout(
            height=500,
            font=dict(family=self.font_family),
            plot_bgcolor=self.bg_color,
            paper_bgcolor=self.bg_color,
            xaxis=dict(
                title="Time",
                showgrid=True,
                gridcolor=self.grid_color,
                showline=True,
                linecolor="black",
                ticks="outside",
                tickcolor="black",
            ),
            yaxis=dict(
                title="Profit [\u20ac]",
                range=y_range,
                showgrid=True,
                gridcolor=self.grid_color,
                zeroline=True,
                zerolinecolor="black",
                zerolinewidth=1,
            ),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            margin=dict(l=60, r=120, t=40, b=40),
        )

        fig.add_annotation(
            text=title,
            xref="paper", yref="paper",
            x=1.10, y=0.5,
            showarrow=False,
            textangle=-90,
            xanchor="center",
            yanchor="middle",
            font=dict(family=self.font_family, size=14),
        )

        return fig

    @staticmethod
    def _profit_helpline_step(y_min: float, y_max: float) -> float:
        """Return a nice round step size so that ~6-9 horizontal help-lines span [y_min, y_max]."""
        span = y_max - y_min
        if span < 1e-6:
            return 50.0
        raw = span / 7.0
        magnitude = 10.0 ** math.floor(math.log10(abs(raw)))
        for m in [1, 2, 2.5, 5, 10, 25, 50]:
            candidate = m * magnitude
            if candidate >= raw:
                return candidate
        return magnitude * 10.0

    # ------------------------------------------------------------------
    # IO
    # ------------------------------------------------------------------

    def _save_figure(self, fig: go.Figure, path: str | Path) -> str:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        export_config = {
            "toImageButtonOptions": {
                "format": self.export_format,
                "filename": path.stem,
                "width": self.export_width,
                "height": self.export_height,
                "scale": self.export_scale,
            }
        }
        fig.write_html(str(path), include_plotlyjs="cdn", config=export_config)
        self.log.info("Saved plot: %s", path)
        return str(path)