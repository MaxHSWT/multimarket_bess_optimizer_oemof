from __future__ import annotations
import logging
import pandas as pd

import src.utils.get_fcr_constraints as get_fcr_constraints
from src.model.market_optimizer import MarketOptimizer


class MultiMarketOptimizer:
    # Daily orchestration layer for the multimarket battery optimization pipeline.

    def __init__(self, cfg, df_FCR, df_DAA, df_IDA, df_IDC, init_soc, df_FCA):
        self.log = logging.getLogger(__name__)
        self.cfg = cfg
        self.optimizer = MarketOptimizer(cfg, df_FCR, df_DAA, df_IDA, df_IDC, init_soc, df_FCA)

        # Config flags
        opt = cfg.optimization
        self.run_fcr = bool(getattr(opt.optimize_FCR, "run_FCR", False))
        self.pfs_fcr = bool(getattr(opt.optimize_FCR, "perfect_foresight", True))
        self.run_daa = bool(getattr(opt.optimize_DAA, "run_DAA", False))
        self.pfs_daa = bool(getattr(opt.optimize_DAA, "perfect_foresight", False))
        self.run_ida = bool(getattr(opt.optimize_IDA, "run_IDA", False))
        self.pfs_ida = bool(getattr(opt.optimize_IDA, "perfect_foresight", False))
        self.run_idc = bool(getattr(opt.optimize_IDC, "run_IDC", False))
        self.pfs_idc = bool(getattr(opt.optimize_IDC, "perfect_foresight", False))

        # Result holders (stay None if stage is not run)
        self.result_fcr = None
        self.result_daa = None
        self.result_ida = None
        self.result_idc = None

        # FCR-derived constraints (initialized neutral = no restrictions for downstream)
        neutral = get_fcr_constraints.neutral_fcr_constraints(
            fcr_index=self.optimizer.df_FCR.index,
            capacity_mwh=float(self.optimizer.capacity),
            max_charge_power_mw=float(self.optimizer.max_charge_power),
            max_discharge_power_mw=float(self.optimizer.max_discharge_power),
        )
        self.min_soc_for_fcr = neutral["min_soc_for_fcr"]
        self.max_soc_for_fcr = neutral["max_soc_for_fcr"]
        self.marketed_pwr_fcr = pd.Series(0.0, index=self.optimizer.df_FCR.index, name="marketed_pwr_fcr")

        # Profit holders
        self.daily_profit_commit = 0.0
        self.daily_profit_horizon = 0.0

        # SoC after committed window
        self.soc_commit = float(init_soc * self.optimizer.capacity)

    # ------------------------------------------------------------------
    # Capacity markets (FCR)
    # ------------------------------------------------------------------

    def run_capacity_markets(self):
        if not self.run_fcr:
            return

        if not self.pfs_fcr:
            self.result_fcr = self.optimizer.optimize_fcr_myopic()
        else:
            self.result_fcr = self.optimizer.optimize_fcr_with_perfect_foresight()

        self.min_soc_for_fcr = self.result_fcr["min_soc_for_fcr"]
        self.max_soc_for_fcr = self.result_fcr["max_soc_for_fcr"]
        self.marketed_pwr_fcr = self.result_fcr["marketed_pwr_fcr"]
        self.daily_profit_commit += float(self.result_fcr["profit_fcr_commit"])
        self.daily_profit_horizon += float(self.result_fcr["profit_fcr_horizon"])

    # ------------------------------------------------------------------
    # Energy markets (DAA -> IDA -> IDC)
    # ------------------------------------------------------------------

    def run_energy_markets(self):

        # --- DAA ---
        if self.run_daa:
            self.result_daa = self.optimizer.optimize_daa(
                min_soc_for_fcr=self.min_soc_for_fcr,
                max_soc_for_fcr=self.max_soc_for_fcr,
                marketed_pwr_fcr=self.marketed_pwr_fcr,
            )
        else:
            self.result_daa = self.optimizer._no_dispatch_daa()
        
        self.daily_profit_commit += float(self.result_daa["profit_daa_commit"])
        self.daily_profit_horizon += float(self.result_daa["profit_daa_horizon"])
        buy_daa_pwr = self.result_daa["buy_daa_pwr"]
        sell_daa_pwr = self.result_daa["sell_daa_pwr"]

        # --- IDA ---
        if self.run_ida:
            self.result_ida = self.optimizer.optimize_ida(
                buy_daa_pwr=buy_daa_pwr,
                sell_daa_pwr=sell_daa_pwr,
                min_soc_for_fcr=self.min_soc_for_fcr,
                max_soc_for_fcr=self.max_soc_for_fcr,
                marketed_pwr_fcr=self.marketed_pwr_fcr,
            )
        else:
            self.result_ida = self.optimizer._no_dispatch_ida(
                buy_daa_pwr=buy_daa_pwr,
                sell_daa_pwr=sell_daa_pwr,
            )
        
        self.daily_profit_commit += float(self.result_ida["profit_ida_commit"])
        self.daily_profit_horizon += float(self.result_ida["profit_ida_horizon"])
        buy_ida_pwr = self.result_ida["buy_ida_pwr"]
        sell_ida_pwr = self.result_ida["sell_ida_pwr"]

        # --- IDC ---
        if self.run_idc:
            self.result_idc = self.optimizer.optimize_idc(
                buy_daa_pwr=buy_daa_pwr,
                sell_daa_pwr=sell_daa_pwr,
                buy_ida_pwr=buy_ida_pwr,
                sell_ida_pwr=sell_ida_pwr,
                min_soc_for_fcr=self.min_soc_for_fcr,
                max_soc_for_fcr=self.max_soc_for_fcr,
                marketed_pwr_fcr=self.marketed_pwr_fcr,
            )
        else:
            self.result_idc = self.optimizer._no_dispatch_idc(
                buy_daa_pwr=buy_daa_pwr,
                sell_daa_pwr=sell_daa_pwr,
                buy_ida_pwr=buy_ida_pwr,
                sell_ida_pwr=sell_ida_pwr,
            )

        self.daily_profit_commit += float(self.result_idc["profit_idc_commit"])
        self.daily_profit_horizon += float(self.result_idc["profit_idc_horizon"])

        # --- Committed SoC (use last stage that was actually run) ---
        if self.run_idc:
            self.soc_commit = float(self.result_idc["soc_idc_commit"])
        elif self.run_ida:
            self.soc_commit = float(self.result_ida["soc_ida_commit"])
        elif self.run_daa:
            self.soc_commit = float(self.result_daa["soc_daa_commit"])

    # ------------------------------------------------------------------
    # Full-day orchestration
    # ------------------------------------------------------------------

    def run_full_day(self):
        self.run_capacity_markets()
        self.run_energy_markets()