import numpy as np
import pandas as pd
from oemof import solph
import pyomo.environ as po
import logging
import src.utils.transform_pd_series as transform_series
import src.utils.transform_pd_dataframes as transform_dataframes
import src.utils.get_fcr_constraints as get_fcr_constraints
import src.utils.get_fca_constraints as get_fca_constraints
import src.utils.calculate_market_dispatch as calculate_market_dispatch




# ------------------------------------------------------------
# Market-level optimization class
# ------------------------------------------------------------
class MarketOptimizer:
    def __init__(self, cfg, df_FCR: pd.DataFrame, df_DAA: pd.DataFrame, df_IDA: pd.DataFrame, df_IDC: pd.DataFrame, initial_soc: float | None = None, df_FCA: pd.DataFrame | None = None):

        # Initialize the optimizer with a Hydra-style configuration.
        self.cfg = cfg
        self.log = logging.getLogger(__name__)

        # Battery configuration
        self.capacity = cfg.battery.capacity    # [MWh]
        self.max_charge_power = cfg.battery.max_charge_power    # [MW]
        self.max_discharge_power = cfg.battery.max_discharge_power  # [MW]
        self.initial_soc = initial_soc if initial_soc is not None else cfg.battery.initial_soc    # [-]
        self.max_cycles = cfg.battery.max_cycles
        self.eff_in = cfg.battery.charge_efficiency
        self.eff_out = cfg.battery.discharge_efficiency
        self.max_marketable_power_fcr = cfg.battery.max_marketable_power_fcr

        # Optimization configuration
        self.solver = cfg.optimization.solver
        self.commit_hours = int(getattr(cfg.optimization.rolling_horizon, "commit_hours", 24))
        self.lookahead_hours = int(getattr(cfg.optimization.rolling_horizon, "lookahead_hours", 24))
        self.run_fcr = bool(getattr(cfg.optimization.optimize_FCR, "run_FCR", False))
        self.pfs_fcr = bool(getattr(cfg.optimization.optimize_FCR, "perfect_foresight", True))
        self.run_daa = bool(getattr(cfg.optimization.optimize_DAA, "run_DAA", False))
        self.pfs_daa = bool(getattr(cfg.optimization.optimize_DAA, "perfect_foresight", False))
        self.run_ida = bool(getattr(cfg.optimization.optimize_IDA, "run_IDA", False))
        self.pfs_ida = bool(getattr(cfg.optimization.optimize_IDA, "perfect_foresight", False))
        self.run_idc = bool(getattr(cfg.optimization.optimize_IDC, "run_IDC", False))
        self.pfs_idc = bool(getattr(cfg.optimization.optimize_IDC, "perfect_foresight", False))
        self.transaction_costs = 0.10 # €/MWh

        # Market Data: Data Frames
        self.df_FCR = df_FCR
        self.df_DAA = df_DAA
        self.df_IDA = df_IDA
        self.df_IDC = df_IDC
        self.df_all_markets = transform_dataframes.transform_all_markets_to_master_df(df_FCR, df_DAA, df_IDA, df_IDC)   # DataFrame with 15-min master index and columns: price_fcr, price_daa, price_ida, price_idc

        # FCA data: Data Frame
        self.df_FCA = df_FCA

        # Dynamic power limits: Series
        if df_FCA is not None:
            fca_limits = get_fca_constraints.derive_fca_power_limits(df_FCA)
            self.max_cha_pwr_fca = fca_limits["max_cha_pwr_fca"]
            self.max_discha_pwr_fca = fca_limits["max_discha_pwr_fca"]
        else:   # Neutral FCA: full technical power available
            fca_limits = get_fca_constraints.neutral_fca_limits(
                index=self.df_all_markets.index,
                max_charge_power_mw=float(self.max_charge_power),
                max_discharge_power_mw=float(self.max_discharge_power),
            )
            self.max_cha_pwr_fca = fca_limits["max_cha_pwr_fca"]
            self.max_discha_pwr_fca = fca_limits["max_discha_pwr_fca"]
    
    def _commit_end_eps(self, timeindex: pd.DatetimeIndex) -> pd.Timestamp:
        window_start = pd.Timestamp(timeindex[0])
        commit_ts = window_start + pd.Timedelta(hours=self.commit_hours)
        return commit_ts - pd.Timedelta(microseconds=1)



    # -------------------------------------------------------------------------------------------------------
    # FCR Myopic Optimization (Class Method) - Capacity trading only - Pyomo only
    # -------------------------------------------------------------------------------------------------------
    def optimize_fcr_myopic(self) -> dict:
        
        # Pure FCR capacity optimization (no energy dispatch -> no oemof environment)
        # Myopic: maximizes FCR revenue without considering opportunity costs from energy markets.
        # Objective: maximize sum_T ( price_fcr[T] * fcr_power[T] )

        log = self.log
        log.info("Building myopic FCR capacity optimization model (Pyomo only)...")


        # --- 1) Input data and Index alignment ---

        fcr_df = self.df_FCR.copy()

        n_blocks = len(fcr_df)                              # number of 4h blocks in the optimization window
        if n_blocks < 1:
            raise ValueError("df_FCR is empty. Need at least 1 FCR block.")
        
        fcr_index = fcr_df.index
        price_fcr = fcr_df["price"].astype(float).values    # length = n_blocks
        BLOCKS = range(n_blocks)                            # 0..n_blocks-1

        # FCR upper bound per 4h block:
        # FCA acts as the effective max power -> PQ-rule (eq. 3.9) applies to FCA-limited power.
        # fcr_ub(B) = pq_factor * min(min_fca_cha_in_block(B), min_fca_discha_in_block(B))
        pq_factor = min(0.8, float(self.max_marketable_power_fcr))
        fca_cha = self.max_cha_pwr_fca
        fca_discha = self.max_discha_pwr_fca
        fca_ub_per_block = []
        for T in BLOCKS:
            block_start = fcr_index[T]
            block_end = block_start + pd.Timedelta(hours=4) - pd.Timedelta(minutes=15)
            mask = (fca_cha.index >= block_start) & (fca_cha.index <= block_end)
            if mask.any():
                min_fca = min(float(fca_cha[mask].min()), float(fca_discha[mask].min()))
                fca_ub_per_block.append(pq_factor * min_fca)
            else:
                # Fallback: use technical power if no FCA data for this block
                fca_ub_per_block.append(pq_factor * float(min(self.max_charge_power, self.max_discharge_power)))


        # --- 2) Build Pyomo model ---

        m = po.ConcreteModel()
        m.BLOCKS = po.Set(initialize=list(BLOCKS)) # Set of n blocks
        
        # Pyomo variable for FCR power per block (per-block upper bound respecting FCA)
        def _fcr_bounds(mm, T):
            return (0.0, fca_ub_per_block[T])
        m.fcr_power = po.Var(m.BLOCKS, domain=po.NonNegativeReals, bounds=_fcr_bounds)

        # Objective: maximize FCR revenue
        revenue_expr = sum(m.fcr_power[T] * float(price_fcr[int(T)]) for T in m.BLOCKS)
        m.obj = po.Objective(expr=revenue_expr, sense=po.maximize)

        # Initial-SOC feasibility: block-0 marketed power must be compatible with starting SoC
        # From min-SOC: initial_soc * capacity >= 0.25 * P  ->  P <= initial_soc * capacity / 0.25
        # From max-SOC: initial_soc * capacity <= capacity - 0.25 * P  ->  P <= (1 - initial_soc) * capacity / 0.25
        fcr_ub_from_min_soc = float(self.initial_soc) * float(self.capacity) / 0.25
        fcr_ub_from_max_soc = (1.0 - float(self.initial_soc)) * float(self.capacity) / 0.25
        fcr_ub_initial = min(fcr_ub_from_min_soc, fcr_ub_from_max_soc)
        m.fcr_initial_soc_limit = po.Constraint(expr=m.fcr_power[0] <= fcr_ub_initial)


        # --- 3) Solve ---

        log.info("Solving pure FCR capacity model with %s solver...", self.solver)
        solver = po.SolverFactory(self.solver)      # Pyomo looks for the solver in the system path to build suitable solver object
        res = solver.solve(m, tee=False)            # Solve model m, tee=False -> no solver output printed to console

        tc = res.solver.termination_condition
        if tc not in (po.TerminationCondition.optimal, po.TerminationCondition.locallyOptimal):
            log.warning("FCR solve ended with %s", tc)



        # --- 4) Extract results ---

        fcr_power_values = [po.value(m.fcr_power[T]) for T in m.BLOCKS]
        marketed_pwr_fcr = pd.Series(fcr_power_values, index=fcr_index, name="fcr_power_MW")

        # Rolling Horizon Commit
        commit_end_eps = self._commit_end_eps(self.df_FCR.index)
        marketed_commit = marketed_pwr_fcr.loc[:commit_end_eps]
        price_fcr_series = pd.Series(price_fcr, index=fcr_index)

        # Revenue from FCR
        profit_fcr_horizon_series = marketed_pwr_fcr * price_fcr_series
        profit_fcr_horizon = float(profit_fcr_horizon_series.sum())
        profit_fcr_commit_series = marketed_commit * price_fcr_series.loc[:commit_end_eps]
        profit_fcr_commit = float(profit_fcr_commit_series.sum())


        # --- 5) Derive constraints for downstream stages (block-wise series) ---

        derived = get_fcr_constraints.derive_fcr_constraints_from_marketed_power(
            marketed_pwr_fcr=marketed_pwr_fcr,
            capacity_mwh=float(self.capacity),
            max_charge_power_mw=float(self.max_charge_power),
            max_discharge_power_mw=float(self.max_discharge_power),
        )

        return {
            "marketed_pwr_fcr": marketed_pwr_fcr,
            "min_soc_for_fcr": derived["min_soc_for_fcr"],
            "max_soc_for_fcr": derived["max_soc_for_fcr"],
            "max_cha_pwr_fcr": derived["max_cha_pwr_fcr"],
            "max_discha_pwr_fcr": derived["max_discha_pwr_fcr"],
            "max_cha_pwr_fca": self.max_cha_pwr_fca,
            "max_discha_pwr_fca": self.max_discha_pwr_fca,
            "profit_fcr_horizon": profit_fcr_horizon,
            "profit_fcr_horizon_series": profit_fcr_horizon_series,
            "profit_fcr_commit": profit_fcr_commit,
            "profit_fcr_commit_series": profit_fcr_commit_series,
        }
    

    def optimize_fcr_with_perfect_foresight(self) -> dict:

        log = self.log
        log.info("Building FCR optimization system...")

        # --- 0) Time index and data alignment ---

        # Oemof-System-Index is 15 min, but FCR blocks are 4h, DAA blocks are configurable (1h or 15min)

        df = self.df_all_markets.copy()

        # Master-Index (15 min):
        timeindex = df.index
        dt_h = (df.index[1] - df.index[0]).total_seconds() / 3600  # time-intervall (0,25h!)
        if abs(dt_h - 0.25) > 1e-9:                                # Sanity Check
            raise ValueError(f"optimize_fcr_with_perfect_foresight expects a 15min master index, got dt_h={dt_h} h.")
        if len(timeindex) % 16 != 0:                               # Sanity Check
            raise ValueError("Length of df_all_markets must be divisible by 16 so FCR can be modeled in 4h blocks.")
        

        # Index-Transformation-Maps for FCR and DAA blocks:
        # Number of block indices
        n_t15 = len(timeindex)          # -> Int: 96 for 24h
        # DAA block size: derived from df_DAA index (4 steps = 1h DAA, 1 step = 15min DAA)
        dt_daa_h = (self.df_DAA.index[1] - self.df_DAA.index[0]).total_seconds() / 3600
        steps_per_daa_block = round(dt_daa_h / dt_h)
        if steps_per_daa_block not in (1, 4):
            raise ValueError(f"DAA frequency must yield 1 (15min) or 4 (1h) steps per block on a 15min master index, got {steps_per_daa_block}.")
        n_daa_blocks = n_t15 // steps_per_daa_block  # -> Int: 24 for 24h (1h DAA) or 96 for 24h (15min DAA)
        n_fcr_blocks = n_t15 // 16      # -> Int:  6 for 24h
        if len(self.df_FCR.index) != n_fcr_blocks:                 # Sanity Check
            raise ValueError(
                f"FCR block mismatch: df_FCR has {len(self.df_FCR.index)} blocks, "
                f"but df_all_markets implies {n_fcr_blocks} 4h blocks."
            )
        # Integer timestep sets for 15min, DAA-blocks, 4h; Better for constraint definitions in Pyomo than using timestamps directly.
        t15 = list(range(n_t15))                    # -> List of 15-min steps: t15 = [0, 1, 2, ..., 95] for 24h
        t1 = list(range(n_daa_blocks))
        t4 = list(range(n_fcr_blocks))
        # Mapping 15min step -> DAA block / FCR 4h block
        t15_to_t1 = {t: t // steps_per_daa_block for t in t15}
        t15_to_t4 = {t: t // 16 for t in t15}       # -> Dict: t15_to_t4 = {0:0, 1:0, ..., 15:0, 16:1, ..., 31:1, ...}
        # Reverse maps
        t1_to_t15 = {h: list(range(steps_per_daa_block * h, steps_per_daa_block * h + steps_per_daa_block)) for h in t1}
        t4_to_t15 = {b: list(range(16 * b, 16 * b + 16)) for b in t4}

        # Timeindexing for cycle limits and rolling horizon
        horizon_hours = n_t15 * dt_h
        t_end_commit = df.index[0] + pd.Timedelta(hours=self.commit_hours)
        t_end_horizon = df.index[0] + pd.Timedelta(hours=horizon_hours)
        T_to_commit = [t for t, ts in enumerate(df.index) if ts < t_end_commit]  # first commit_hours
        T_to_horizon_without_commit = [t for t, ts in enumerate(df.index) if (ts >= t_end_commit and ts < t_end_horizon)]  # Indexes of the lookahead after the first 24h

        # Prices
        price_daa = df["price_daa"].astype(float)        # can stay in 15-min resolution, will be mapped by constraints
        price_ida = df["price_ida"].astype(float)
        price_idc = df["price_idc"].astype(float)
        price_fcr = self.df_FCR["price"].astype(float)   # easier for calculating fcr revenue -> pyomo variable!



        # --- 1) Build energy system ---

        es = solph.EnergySystem(timeindex=timeindex, infer_last_interval=True)
        b_el = solph.Bus(label="electricity")

        # Variable DAA buy (forecast)
        source_daa = solph.components.Source(
            label="source_daa",
            outputs={
                b_el: solph.Flow(
                    nominal_value=self.max_charge_power,        # MW
                    variable_costs=(price_daa + self.transaction_costs).values             # €/MWh
                )
            } if self.run_daa else {}
        )
        # Variable DAA sell (forecast)
        sink_daa = solph.components.Sink(
            label="sink_daa",
            inputs={
                b_el: solph.Flow(
                    nominal_value=self.max_discharge_power,     # MW
                    variable_costs=-(price_daa - self.transaction_costs).values            # €/MWh (revenue)
                )
            } if self.run_daa else {}
        )

        # Variable IDA buy (forecast)
        source_ida = solph.components.Source(
            label="source_ida",
            outputs={
                b_el: solph.Flow(
                    nominal_value=self.max_charge_power,        # MW
                    variable_costs=(price_ida + self.transaction_costs).values             # €/MWh
                )
            } if self.run_ida else {}
        ) 
        # Variable IDA sell (forecast)
        sink_ida = solph.components.Sink(
            label="sink_ida",
            inputs={
                b_el: solph.Flow(
                    nominal_value=self.max_discharge_power,     # MW
                    variable_costs=-(price_ida - self.transaction_costs).values            # €/MWh (revenue)
                )
            } if self.run_ida else {}
        )

        # Variable IDC buy (forecast)
        source_idc = solph.components.Source(
            label="source_idc",
            outputs={
                b_el: solph.Flow(
                    nominal_value=self.max_charge_power,        # MW
                    variable_costs=(price_idc + self.transaction_costs).values             # €/MWh
                )
            } if self.run_idc else {}
        )
        # Variable IDC sell (forecast)
        sink_idc = solph.components.Sink(
            label="sink_idc",
            inputs={
                b_el: solph.Flow(
                    nominal_value=self.max_discharge_power,     # MW
                    variable_costs=-(price_idc - self.transaction_costs).values            # €/MWh (revenue)
                )
            } if self.run_idc else {}
        )

        # Battery storage
        # FCA limits applied as max sequences on battery input/output flows
        fca_max_in_seq = transform_series.make_flow_max_sequences(
            timeindex=timeindex,
            max_power_mw=self.max_cha_pwr_fca,
            nominal_power_mw=self.max_charge_power,
        )
        fca_max_out_seq = transform_series.make_flow_max_sequences(
            timeindex=timeindex,
            max_power_mw=self.max_discha_pwr_fca,
            nominal_power_mw=self.max_discharge_power,
        )
        battery = solph.components.GenericStorage(
            label="battery",
            nominal_storage_capacity=self.capacity,                                                     # MWh
            inputs={b_el: solph.Flow(nominal_value=self.max_charge_power, max=fca_max_in_seq)},         # MW, limited by FCA
            outputs={b_el: solph.Flow(nominal_value=self.max_discharge_power, max=fca_max_out_seq)},    # MW, limited by FCA
            loss_rate=0.0,
            initial_storage_level=self.initial_soc,
            min_storage_level=0.0,
            max_storage_level=1.0,                             # static bounds, dynamic bounds will be implemented as constraints
            balanced=False,
            inflow_conversion_factor=self.eff_in,
            outflow_conversion_factor=self.eff_out,
        )

        # Add components to energy system
        es.add(b_el, battery)
        if self.run_daa: es.add(source_daa, sink_daa)
        if self.run_ida: es.add(source_ida, sink_ida)
        if self.run_idc: es.add(source_idc, sink_idc)


        # --- 2) Build model ---

        log.info("Creating FCR model...")
        model = solph.Model(es)
        T = list(model.TIMESTEPS)   # oemof timeindex as list of timestamps (15-min resolution), [0, 1, 2, ..., 95] for 24h, oemof/pyomo can not work with pandas DatetimeIndex directly for constraint definitions.


        # --- 3) Add FCR decision variable and constraints ---

        # (1) FCR block variable
        blk_fcr = po.Block()
        model.add_component("fcr_block", blk_fcr)
        blk_fcr.FCR_BLOCKS = po.Set(initialize=t4)
        # Defining fcr_power as a pyomo variable indexed by FCR blocks, with non-negativity
        blk_fcr.fcr_power = po.Var(blk_fcr.FCR_BLOCKS, domain=po.NonNegativeReals)
        # FCR upper bound per 4h block:
        # FCA acts as the effective max power -> PQ-rule (eq. 3.9) applies to FCA-limited power.
        # fcr_power(B) <= pq_factor * min(min_fca_cha_in_block(B), min_fca_discha_in_block(B))
        pq_factor = min(float(self.max_marketable_power_fcr), 0.8)
        fca_cha_reindexed = self.max_cha_pwr_fca.reindex(timeindex, method="ffill")
        fca_discha_reindexed = self.max_discha_pwr_fca.reindex(timeindex, method="ffill")
        fca_ub_per_block = {}
        for B in t4:
            t_steps = t4_to_t15[B]
            min_fca_cha = float(fca_cha_reindexed.iloc[t_steps].min())
            min_fca_discha = float(fca_discha_reindexed.iloc[t_steps].min())
            fca_ub_per_block[B] = pq_factor * min(min_fca_cha, min_fca_discha)
        # Applying upper bound constraint for each FCR block
        def _fcr_ub_rule(b, B):
            return b.fcr_power[B] <= fca_ub_per_block[B]
        blk_fcr.fcr_ub = po.Constraint(blk_fcr.FCR_BLOCKS, rule=_fcr_ub_rule)

        # (2) Dynamic SoC bounds caused by FCR
        # Attention: storage_content is indexed with the state AFTER timestep t -> use t+1 and define storage_content[0] extra
        # min_soc_for_fcr(T) = 0.25 * fcr_power(T)  [MWh]  (referring to PQ-Document eq. 3.8)
        def _soc_min_rule(m, t):
            B = t15_to_t4[t]     # Find the corresponding FCR block for this 15-min timestep
            return m.GenericStorageBlock.storage_content[battery, t + 1] >= 0.25 * blk_fcr.fcr_power[B]
        # max_soc_for_fcr(T) = capacity - 0.25 * fcr_power(T)  [MWh]  (referring to PQ-Document eq. 3.7)
        def _soc_max_rule(m, t):
            B = t15_to_t4[t]    
            return m.GenericStorageBlock.storage_content[battery, t + 1] <= self.capacity - 0.25 * blk_fcr.fcr_power[B]
        model.fcr_soc_min = po.Constraint(T, rule=_soc_min_rule)
        model.fcr_soc_max = po.Constraint(T, rule=_soc_max_rule)
        # Initial-state SoC constraint: storage_content[battery, 0] must also respect block-0 FCR commitment
        B0 = t4[0]
        model.fcr_soc_min_initial = po.Constraint(expr=model.GenericStorageBlock.storage_content[battery, 0] >= 0.25 * blk_fcr.fcr_power[B0])
        model.fcr_soc_max_initial = po.Constraint(expr=model.GenericStorageBlock.storage_content[battery, 0] <= self.capacity - 0.25 * blk_fcr.fcr_power[B0])

        # (3) Dynamic charge/discharge limits due to reserved FCR capacity (respecting FCA limits)
        # The headroom constraint uses the FCA-limited power as the effective upper bound per timestep.
        # P_charge(t) + fcr_power(B) <= fca_cha_limit(t)
        fca_cha_vals = self.max_cha_pwr_fca.reindex(timeindex, method="ffill").values
        fca_discha_vals = self.max_discha_pwr_fca.reindex(timeindex, method="ffill").values
        def _charge_limit_rule(m, t):
            B = t15_to_t4[t]
            return m.flow[b_el, battery, t] + blk_fcr.fcr_power[B] <= float(fca_cha_vals[t])
        # P_discharge(t) + fcr_power(B) <= fca_discha_limit(t)
        def _discharge_limit_rule(m, t):
            B = t15_to_t4[t]
            return m.flow[battery, b_el, t] + blk_fcr.fcr_power[B] <= float(fca_discha_vals[t])
        model.fcr_charge_headroom = po.Constraint(T, rule=_charge_limit_rule)
        model.fcr_discharge_headroom = po.Constraint(T, rule=_discharge_limit_rule)

        # (4) DAA block constancy: force DAA trades to be equal within each DAA block. Skipped for 15min DAA (steps_per_daa_block == 1), where each 15-min slot is already its own block.
        if self.run_daa and steps_per_daa_block > 1:
            blk_daa_block = po.Block()
            model.add_component("daa_hour_block", blk_daa_block)
            def _daa_buy_block_rule(b, h, k):
                idxs = t1_to_t15[h]
                if k == 0:
                    return po.Constraint.Skip
                return model.flow[source_daa, b_el, idxs[k]] == model.flow[source_daa, b_el, idxs[0]]
            def _daa_sell_block_rule(b, h, k):
                idxs = t1_to_t15[h]
                if k == 0:
                    return po.Constraint.Skip
                return model.flow[b_el, sink_daa, idxs[k]] == model.flow[b_el, sink_daa, idxs[0]]
            blk_daa_block.buy_constancy = po.Constraint([(h, k) for h in t1 for k in range(steps_per_daa_block)], rule=_daa_buy_block_rule)
            blk_daa_block.sell_constancy = po.Constraint([(h, k) for h in t1 for k in range(steps_per_daa_block)], rule=_daa_sell_block_rule)


        # (5) Cycle limits
        blk_cycles = po.Block()
        model.add_component("fcr_cycle_limit_block", blk_cycles)
        max_total_charge_per_day = self.max_cycles * self.capacity  # limit per 24h (commit window) # MWh, for 2 cycles/day: 2*capacity
        lookahead_hours_without_commit = max(0.0, horizon_hours - self.commit_hours)
        max_total_charge_in_lookahead = self.max_cycles * self.capacity * (lookahead_hours_without_commit / 24.0)
        def _max_cycles_day_rule(b):
            return sum(model.flow[b_el, battery, t] * dt_h for t in T_to_commit) <= max_total_charge_per_day
        def _max_cycles_lookahead_rule(b):     # Rule for the maximum cycles in the lookahead (after the commit period)
            if lookahead_hours_without_commit <= 0 or len(T_to_horizon_without_commit) == 0:    # Skip the constraint if there is no lookahead after the commit period
                return po.Constraint.Skip
            expr = sum(model.flow[b_el, battery, t] * dt_h for t in T_to_horizon_without_commit) <= max_total_charge_in_lookahead
            return expr
        blk_cycles.max_cycles_per_day = po.Constraint(rule=_max_cycles_day_rule)
        blk_cycles.max_cycles_per_horizon = po.Constraint(rule=_max_cycles_lookahead_rule)

        # (6) Prevent simultaneous buy and sell in DAA / IDA / IDC. Caution: Makes the problem a Mixed-Integer-Problem!
        # DAA
        if self.run_daa:
            blk_daa_mode = po.Block()
            model.add_component("daa_no_simul_block", blk_daa_mode)
            # Binary mode: 1 = buy allowed, 0 = sell allowed -> binary / integer variable
            blk_daa_mode.y_buy = po.Var(T, domain=po.Binary)    # Decision variable with values {0, 1} for every timestep t. x -> usually continuous variables, y -> usually discrete variables (integer, binary)
            def _daa_buy_limit_rule(b, t):      # Buy logic: buying is only allowed if y_buy[t] == 1
                return model.flow[source_daa, b_el, t] <= self.max_charge_power * b.y_buy[t]
            def _daa_sell_limit_rule(b, t):     # Sell logic: selling is only allowed if y_buy[t] == 0
                return model.flow[b_el, sink_daa, t] <= self.max_discharge_power * (1 - b.y_buy[t])
            blk_daa_mode.buy_limit = po.Constraint(T, rule=_daa_buy_limit_rule)
            blk_daa_mode.sell_limit = po.Constraint(T, rule=_daa_sell_limit_rule) 
        # IDA
        if self.run_ida:
            blk_ida_mode = po.Block()
            model.add_component("ida_no_simul_block", blk_ida_mode)
            blk_ida_mode.y_buy = po.Var(T, domain=po.Binary)
            def _ida_buy_limit_rule(b, t):      
                return model.flow[source_ida, b_el, t] <= self.max_charge_power * b.y_buy[t]
            def _ida_sell_limit_rule(b, t):     
                return model.flow[b_el, sink_ida, t] <= self.max_discharge_power * (1 - b.y_buy[t])
            blk_ida_mode.buy_limit = po.Constraint(T, rule=_ida_buy_limit_rule)
            blk_ida_mode.sell_limit = po.Constraint(T, rule=_ida_sell_limit_rule) 
        # IDC
        if self.run_idc:
            blk_idc_mode = po.Block()
            model.add_component("idc_no_simul_block", blk_idc_mode)
            blk_idc_mode.y_buy = po.Var(T, domain=po.Binary)   
            def _idc_buy_limit_rule(b, t):      
                return model.flow[source_idc, b_el, t] <= self.max_charge_power * b.y_buy[t]
            def _idc_sell_limit_rule(b, t): 
                return model.flow[b_el, sink_idc, t] <= self.max_discharge_power * (1 - b.y_buy[t])
            blk_idc_mode.buy_limit = po.Constraint(T, rule=_idc_buy_limit_rule)
            blk_idc_mode.sell_limit = po.Constraint(T, rule=_idc_sell_limit_rule) 

        # (7) Add FCR revenue to objective
        # solph minimizes system costs. FCR revenue must therefore be SUBTRACTED.
        fcr_revenue_expr = sum(blk_fcr.fcr_power[B] * float(price_fcr.iloc[B]) for B in t4)
        base_objective_expr = model.objective.expr
        model.objective.deactivate()
        model.total_objective = po.Objective(
            expr=base_objective_expr - fcr_revenue_expr, # oemof tries to minimize system costs
            sense=po.minimize,
        )


        # --- 4) Solve ---

        log.info(f"Solving FCR with {self.solver} solver...")
        model.solve(solver=self.solver, solve_kwargs={"tee": False})


        # --- 5) Extract FCR results ---
        
        # FCR-Power-Blocks have Indices [0, 1, ..., 5] and are hence not compatible with oemof-result-handeling
        marketed_pwr_fcr = pd.Series(
            [po.value(blk_fcr.fcr_power[B]) for B in t4],       # get values from pyomo variable and build series
            index=self.df_FCR.index,                            # Frequency = 4h, Indices = 00:00, 04:00, ...
            name="marketed_pwr_fcr",
        )
        # Remove custom 4h block before oemof result processing
        model.del_component(blk_fcr)


        # --- 6) Extract all other results ---

        log.info("Processing FCR results...")
        results = solph.processing.results(model)
        zero_series = pd.Series(0.0, index=timeindex)

        # Energy Flows:

        # Charging / Discharging
        cha_fcr_pwr = results[(b_el, battery)]["sequences"]["flow"]         # MW
        cha_fcr = cha_fcr_pwr * dt_h                                        # MWh
        discha_fcr_pwr = results[(battery, b_el)]["sequences"]["flow"]      # MW
        discha_fcr = discha_fcr_pwr * dt_h                                  # MWh
        # SoC
        soc_fcr_horizon = results[(battery, None)]["sequences"]["storage_content"]  # MWh

        # DAA buy / sell
        if self.run_daa:
            buy_daa_pwr_fcr = results[(source_daa, b_el)]["sequences"]["flow"]  # MW
            sell_daa_pwr_fcr = results[(b_el, sink_daa)]["sequences"]["flow"]   # MW
        else:
            buy_daa_pwr_fcr = zero_series.copy()
            sell_daa_pwr_fcr = zero_series.copy()
        buy_daa_fcr = buy_daa_pwr_fcr * dt_h                                # MWh
        sell_daa_fcr = sell_daa_pwr_fcr * dt_h                              # MWh

        # IDA buy / sell
        if self.run_ida:
            buy_ida_pwr_fcr = results[(source_ida, b_el)]["sequences"]["flow"]  # MW
            sell_ida_pwr_fcr = results[(b_el, sink_ida)]["sequences"]["flow"]   # MW
        else:
            buy_ida_pwr_fcr = zero_series.copy()
            sell_ida_pwr_fcr = zero_series.copy()
        buy_ida_fcr = buy_ida_pwr_fcr * dt_h                                # MWh
        sell_ida_fcr = sell_ida_pwr_fcr * dt_h                              # MWh

        # IDC buy / sell
        if self.run_idc:
            buy_idc_pwr_fcr = results[(source_idc, b_el)]["sequences"]["flow"]  # MW
            sell_idc_pwr_fcr = results[(b_el, sink_idc)]["sequences"]["flow"]   # MW
        else:
            buy_idc_pwr_fcr = zero_series.copy()
            sell_idc_pwr_fcr = zero_series.copy()
        buy_idc_fcr = buy_idc_pwr_fcr * dt_h                                # MWh
        sell_idc_fcr = sell_idc_pwr_fcr * dt_h                              # MWh

        # Rolling Horizon Commit
        commit_end_eps = self._commit_end_eps(df.index)
        commit_end_eps_15 = self._commit_end_eps(timeindex)
        marketed_commit = marketed_pwr_fcr.loc[:commit_end_eps]
        price_fcr_series = price_fcr.reindex(marketed_pwr_fcr.index)
        soc_fcr_commit = float(soc_fcr_horizon.loc[:commit_end_eps_15].iloc[-1])

        # Revenue from FCR
        profit_fcr_horizon_series = marketed_pwr_fcr * price_fcr_series
        profit_fcr_horizon = float(profit_fcr_horizon_series.sum())
        profit_fcr_commit_series = marketed_commit * price_fcr_series.loc[:commit_end_eps_15]
        profit_fcr_commit = float(profit_fcr_commit_series.sum())

        # Pot. revenue from energy trading
        profit_daa_horizon = float(((sell_daa_fcr - buy_daa_fcr) * price_daa.reindex(timeindex)).sum())
        profit_daa_commit = float((((sell_daa_fcr - buy_daa_fcr) * price_daa.reindex(timeindex)).loc[:commit_end_eps_15]).sum())
        profit_ida_horizon = float(((sell_ida_fcr - buy_ida_fcr) * price_ida.reindex(timeindex)).sum())
        profit_ida_commit = float((((sell_ida_fcr - buy_ida_fcr) * price_ida.reindex(timeindex)).loc[:commit_end_eps_15]).sum())
        profit_idc_horizon = float(((sell_idc_fcr - buy_idc_fcr) * price_idc.reindex(timeindex)).sum())
        profit_idc_commit = float((((sell_idc_fcr - buy_idc_fcr) * price_idc.reindex(timeindex)).loc[:commit_end_eps_15]).sum())


        # --- 7) Derive constraints for downstream stages (block-wise series?) ---
        derived = get_fcr_constraints.derive_fcr_constraints_from_marketed_power(
            marketed_pwr_fcr=marketed_pwr_fcr,
            capacity_mwh=float(self.capacity),
            max_charge_power_mw=float(self.max_charge_power),
            max_discharge_power_mw=float(self.max_discharge_power),
        )


        return {
            "marketed_pwr_fcr": marketed_pwr_fcr,
            "min_soc_for_fcr": derived["min_soc_for_fcr"],
            "max_soc_for_fcr": derived["max_soc_for_fcr"],
            "max_cha_pwr_fcr": derived["max_cha_pwr_fcr"],
            "max_discha_pwr_fcr": derived["max_discha_pwr_fcr"],
            "max_cha_pwr_fca": self.max_cha_pwr_fca,
            "max_discha_pwr_fca": self.max_discha_pwr_fca,
            "profit_fcr_horizon": profit_fcr_horizon,
            "profit_fcr_horizon_series": profit_fcr_horizon_series,
            "profit_fcr_commit": profit_fcr_commit,
            "profit_fcr_commit_series": profit_fcr_commit_series,
            "profit_daa_horizon": profit_daa_horizon,
            "profit_daa_commit": profit_daa_commit,
            "profit_ida_horizon": profit_ida_horizon,
            "profit_ida_commit": profit_ida_commit,
            "profit_idc_horizon": profit_idc_horizon,
            "profit_idc_commit": profit_idc_commit,
            "cha_fcr": cha_fcr,
            "cha_fcr_pwr": cha_fcr_pwr,
            "discha_fcr": discha_fcr,
            "discha_fcr_pwr": discha_fcr_pwr,
            "buy_daa_fcr": buy_daa_fcr,
            "buy_daa_pwr_fcr": buy_daa_pwr_fcr,
            "sell_daa_fcr": sell_daa_fcr,
            "sell_daa_pwr_fcr": sell_daa_pwr_fcr,
            "buy_ida_fcr": buy_ida_fcr,
            "buy_ida_pwr_fcr": buy_ida_pwr_fcr,
            "sell_ida_fcr": sell_ida_fcr,
            "sell_ida_pwr_fcr": sell_ida_pwr_fcr,
            "buy_idc_fcr": buy_idc_fcr,
            "buy_idc_pwr_fcr": buy_idc_pwr_fcr,
            "sell_idc_fcr": sell_idc_fcr,
            "sell_idc_pwr_fcr": sell_idc_pwr_fcr,
            "soc_fcr_horizon": soc_fcr_horizon,
            "soc_fcr_commit": soc_fcr_commit,
        }



    # ------------------------------------------------------------
    # Day-Ahead Optimization (Class Method)
    # ------------------------------------------------------------

    def optimize_daa(self, min_soc_for_fcr: pd.Series, max_soc_for_fcr: pd.Series, marketed_pwr_fcr: pd.Series):
    
        log = self.log
        log.info("Building DAA optimization system...")


        #--- 0) Time index and data alignment ---

        df = self.df_all_markets.copy()

        # Master-Index (15 min):
        timeindex = df.index
        dt_h = (df.index[1] - df.index[0]).total_seconds() / 3600  # time-intervall (0,25h!)
        if abs(dt_h - 0.25) > 1e-9:                                # Sanity Check
            raise ValueError(f"optimize_daa expects a 15min master index, got dt_h={dt_h} h.")

        # DAA block size: derived from df_DAA index (4 steps = 1h DAA, 1 step = 15min DAA)
        dt_daa_h = (self.df_DAA.index[1] - self.df_DAA.index[0]).total_seconds() / 3600
        steps_per_daa_block = round(dt_daa_h / dt_h)
        if steps_per_daa_block not in (1, 4):
            raise ValueError(f"DAA frequency must yield 1 (15min) or 4 (1h) steps per block on a 15min master index, got {steps_per_daa_block}.")
        if steps_per_daa_block > 1 and len(timeindex) % steps_per_daa_block != 0:  # Sanity Check
            raise ValueError(f"Length of df_all_markets ({len(timeindex)}) must be divisible by {steps_per_daa_block} for DAA block modeling.")
        
        # Index-Transformation-Maps for DAA blocks:
        # Number of block indices
        n_t15 = len(timeindex)          # -> Int: 96 for 24h
        n_daa_blocks = n_t15 // steps_per_daa_block  # -> Int: 24 for 24h (1h DAA) or 96 for 24h (15min DAA)
        if len(self.df_DAA.index) != n_daa_blocks:                 # Sanity Check
            raise ValueError(
                f"DAA block mismatch: df_DAA has {len(self.df_DAA.index)} blocks, "
                f"but df_all_markets implies {n_daa_blocks} {'15min' if steps_per_daa_block == 1 else '1h'} blocks."
            )
        # Integer timestep sets for 15min and DAA-blocks; Better for constraint definitions in Pyomo than using timestamps directly.
        t15 = list(range(n_t15))                    # -> List of 15-min steps: t15 = [0, 1, 2, ..., 95] for 24h
        t1 = list(range(n_daa_blocks))
        # Mapping 15min step -> DAA block
        t15_to_t1 = {t: t // steps_per_daa_block for t in t15}
        # Reverse map
        t1_to_t15 = {h: list(range(steps_per_daa_block * h, steps_per_daa_block * h + steps_per_daa_block)) for h in t1}

        # Timeindexing for cycle limits if rolling horizon is active
        horizon_hours = n_t15 * dt_h
        t_end_commit = df.index[0] + pd.Timedelta(hours=self.commit_hours)
        t_end_horizon = df.index[0] + pd.Timedelta(hours=horizon_hours)
        T_to_commit = [t for t, ts in enumerate(df.index) if ts < t_end_commit]   # Indexes of the first 24h in the DAA time series
        T_to_horizon_without_commit = [t for t, ts in enumerate(df.index) if (ts >= t_end_commit and ts < t_end_horizon)]  # Indexes of the lookahead after the first 24h

        # Prices
        price_daa = df["price_daa"].astype(float)        # can stay in 15-min resolution, will be mapped by constraints
        price_ida = df["price_ida"].astype(float)
        price_idc = df["price_idc"].astype(float)


        #--- 1) Create energy system ---
        log.info("Building DAA energy system...")
        es = solph.EnergySystem(timeindex=timeindex, infer_last_interval=True)
        b_el = solph.Bus(label="electricity")

        # Variable DAA buy (on time)
        source_daa = solph.components.Source(
            label="source_daa",
            outputs={
                b_el: solph.Flow(
                    nominal_value=self.max_charge_power,        # MW
                    variable_costs=(price_daa + self.transaction_costs).values)            # €/MWh
            },
        )

        # Variable DAA sell (on time)
        sink_daa = solph.components.Sink(
            label="sink_daa",
            inputs={
                b_el: solph.Flow(
                    nominal_value=self.max_discharge_power,     # MW
                    variable_costs=-(price_daa - self.transaction_costs).values)           # €/MWh
            },
        )

        # Variable IDA buy (forecast)
        source_ida = solph.components.Source(
            label="source_ida",
            outputs={
                b_el: solph.Flow(
                    nominal_value=self.max_charge_power,        # MW
                    variable_costs=(price_ida + self.transaction_costs).values             # €/MWh
                )
            } if self.pfs_daa and self.run_ida else {}
        )
        # Variable IDA sell (forecast)
        sink_ida = solph.components.Sink(
            label="sink_ida",
            inputs={
                b_el: solph.Flow(
                    nominal_value=self.max_discharge_power,     # MW
                    variable_costs=-(price_ida - self.transaction_costs).values            # €/MWh (revenue)
                )
            } if self.pfs_daa and self.run_ida else {}
        )

        # Variable IDC buy (forecast)
        source_idc = solph.components.Source(
            label="source_idc",
            outputs={
                b_el: solph.Flow(
                    nominal_value=self.max_charge_power,        # MW
                    variable_costs=(price_idc + self.transaction_costs).values             # €/MWh
                )
            } if self.pfs_daa and self.run_idc else {}
        )
        # Variable IDC sell (forecast)
        sink_idc = solph.components.Sink(
            label="sink_idc",
            inputs={
                b_el: solph.Flow(
                    nominal_value=self.max_discharge_power,     # MW
                    variable_costs=-(price_idc - self.transaction_costs).values            # €/MWh (revenue)
                )
            } if self.pfs_daa and self.run_idc else {}
        )

        # Battery storage:
        # Defining SoC limits by FCR and Power limits by FCR and FCA (data allignment: pd.Series to sequences as required by oemof)
        min_level_seq, max_level_seq = transform_series.make_storage_level_sequences(
            timeindex=df.index,
            min_soc_mwh=min_soc_for_fcr,
            max_soc_mwh=max_soc_for_fcr,
            capacity_mwh=self.capacity,
        )
        # Compute active power limits: FCA ceiling minus FCR reservation
        active_limits = get_fca_constraints.compute_active_power_limits(
            max_cha_pwr_fca=self.max_cha_pwr_fca,
            max_discha_pwr_fca=self.max_discha_pwr_fca,
            marketed_pwr_fcr=marketed_pwr_fcr,
        )
        max_cha_pwr_active = active_limits["max_cha_pwr_active"]
        max_discha_pwr_active = active_limits["max_discha_pwr_active"]
        max_in_seq = transform_series.make_flow_max_sequences(
            timeindex=df.index,
            max_power_mw=max_cha_pwr_active,
            nominal_power_mw=self.max_charge_power,
        )
        max_out_seq = transform_series.make_flow_max_sequences(
            timeindex=df.index,
            max_power_mw=max_discha_pwr_active,
            nominal_power_mw=self.max_discharge_power,
        )
        # Defining the battery
        battery = solph.components.GenericStorage(
            label="battery",
            nominal_storage_capacity=self.capacity,
            inputs={b_el: solph.Flow(nominal_value=self.max_charge_power, max=max_in_seq)}, 
            outputs={b_el: solph.Flow(nominal_value=self.max_discharge_power, max=max_out_seq)},
            loss_rate=0.0,
            initial_storage_level=self.initial_soc,
            min_storage_level=min_level_seq,
            max_storage_level=max_level_seq,
            balanced=False,
            inflow_conversion_factor=self.eff_in,
            outflow_conversion_factor=self.eff_out,
        )

        # Add all components
        es.add(b_el, source_daa, sink_daa, battery)
        if self.pfs_daa:
            if self.run_ida:
                es.add(source_ida, sink_ida)
            if self.run_idc:
                es.add(source_idc, sink_idc)


        #--- 2) Build model ---
        log.info("Creating model...")
        model = solph.Model(es)          # Now all important pyomo variables exist (e.g. model.flow, model.storage_content, ...)
        T = list(model.TIMESTEPS)


        #--- 3) Add custom constraints ---

        # (1) Cycle limit
        blk_cycles = po.Block()
        model.add_component("daa_cycle_limit_block", blk_cycles)
        max_total_charge_per_day = self.max_cycles * self.capacity  # Charging limit per day
        lookahead_hours_without_commit = max(0.0, horizon_hours - self.commit_hours)
        max_total_charge_in_lookahead = self.max_cycles * self.capacity * (lookahead_hours_without_commit / 24.0)
        def _max_cycles_day_rule(b):        # Rule for the maximum cycles per day (the block must be passed as argument, Pyomo-specific)
            expr = sum(model.flow[b_el, battery, t] * dt_h for t in T_to_commit) <= max_total_charge_per_day
            return expr
        def _max_cycles_lookahead_rule(b):     # Rule for the maximum cycles in the lookahead (after the commit period)
            if lookahead_hours_without_commit <= 0 or len(T_to_horizon_without_commit) == 0:    # Skip the constraint if there is no lookahead after the commit period
                return po.Constraint.Skip
            expr = sum(model.flow[b_el, battery, t] * dt_h for t in T_to_horizon_without_commit) <= max_total_charge_in_lookahead
            return expr
        blk_cycles.max_cycles_per_day = po.Constraint(rule = _max_cycles_day_rule)             # adds the constraints to the block
        blk_cycles.max_cycles_lookahead = po.Constraint(rule = _max_cycles_lookahead_rule)

        # (2) Prevent simultaneous buy and sell in DAA/IDA/IDC. Caution: Makes the problem a Mixed-Integer-Problem!
        # DAA
        blk_daa_mode = po.Block()
        model.add_component("daa_no_simul_block", blk_daa_mode)
        # Binary mode: 1 = buy allowed, 0 = sell allowed -> binary / integer variable
        blk_daa_mode.y_buy = po.Var(T, domain=po.Binary)    # Decision variable with values {0, 1} for every timestep t. x -> usually continuous variables, y -> usually discrete variables (integer, binary)
        def _daa_buy_limit_rule(b, t):      # Buy logic: buying is only allowed if y_buy[t] == 1
            return model.flow[source_daa, b_el, t] <= self.max_charge_power * b.y_buy[t]
        def _daa_sell_limit_rule(b, t):     # Sell logic: selling is only allowed if y_buy[t] == 0
            return model.flow[b_el, sink_daa, t] <= self.max_discharge_power * (1 - b.y_buy[t])
        blk_daa_mode.buy_limit = po.Constraint(T, rule=_daa_buy_limit_rule)
        blk_daa_mode.sell_limit = po.Constraint(T, rule=_daa_sell_limit_rule)  
        # IDA + IDC (if perfect foresight)
        if self.pfs_daa and self.run_ida:
            # IDA
            blk_ida_mode = po.Block()
            model.add_component("ida_no_simul_block", blk_ida_mode)
            blk_ida_mode.y_buy = po.Var(T, domain=po.Binary)
            def _ida_buy_limit_rule(b, t):      
                return model.flow[source_ida, b_el, t] <= self.max_charge_power * b.y_buy[t]
            def _ida_sell_limit_rule(b, t):     
                return model.flow[b_el, sink_ida, t] <= self.max_discharge_power * (1 - b.y_buy[t])
            blk_ida_mode.buy_limit = po.Constraint(T, rule=_ida_buy_limit_rule)
            blk_ida_mode.sell_limit = po.Constraint(T, rule=_ida_sell_limit_rule)
        if self.pfs_daa and self.run_idc:
            # IDC
            blk_idc_mode = po.Block()
            model.add_component("idc_no_simul_block", blk_idc_mode)
            blk_idc_mode.y_buy = po.Var(T, domain=po.Binary)   
            def _idc_buy_limit_rule(b, t):      
                return model.flow[source_idc, b_el, t] <= self.max_charge_power * b.y_buy[t]
            def _idc_sell_limit_rule(b, t): 
                return model.flow[b_el, sink_idc, t] <= self.max_discharge_power * (1 - b.y_buy[t])
            blk_idc_mode.buy_limit = po.Constraint(T, rule=_idc_buy_limit_rule)
            blk_idc_mode.sell_limit = po.Constraint(T, rule=_idc_sell_limit_rule)

        # (3) DAA block constancy: force DAA trades to be equal within each DAA block. Skipped for 15min DAA (steps_per_daa_block == 1), where each 15-min slot is already its own block.
        if steps_per_daa_block > 1:
            blk_daa_block = po.Block()
            model.add_component("daa_hour_block", blk_daa_block)
            def _daa_buy_block_rule(b, h, k):
                idxs = t1_to_t15[h]
                if k == 0:
                    return po.Constraint.Skip
                return model.flow[source_daa, b_el, idxs[k]] == model.flow[source_daa, b_el, idxs[0]]
            def _daa_sell_block_rule(b, h, k):
                idxs = t1_to_t15[h]
                if k == 0:
                    return po.Constraint.Skip
                return model.flow[b_el, sink_daa, idxs[k]] == model.flow[b_el, sink_daa, idxs[0]]
            blk_daa_block.buy_constancy = po.Constraint([(h, k) for h in t1 for k in range(steps_per_daa_block)], rule=_daa_buy_block_rule)
            blk_daa_block.sell_constancy = po.Constraint([(h, k) for h in t1 for k in range(steps_per_daa_block)], rule=_daa_sell_block_rule)


        #--- 4) Solve ---

        log.info(f"Solving with {self.solver} solver...")
        model.solve(solver=self.solver, solve_kwargs={"tee": False})


        #--- 5) Extract results ---

        log.info("Processing DAA results...")
        results = solph.processing.results(model)   # gives back the results as a python dictionary holding pandas Series for scalar values and pandas DataFrames for all nodes and flows between them
        zero_series = pd.Series(0.0, index=timeindex)
        
        # Extract sequences as pd.DataFrames
        seq_charge = results[(b_el, battery)]["sequences"]
        seq_discharge = results[(battery, b_el)]["sequences"]
        seq_storage = results[(battery, None)]["sequences"]

        # Extract flows as pd.Series; IMPORTANT: Flows are always in MW!
        cha_daa_pwr = seq_charge["flow"]                                  # [MW] -> charging power from bus into battery per interval
        discha_daa_pwr = seq_discharge["flow"]                            # [MW] -> discharging power from battery to bus per interval
        buy_daa_pwr = results[(source_daa, b_el)]["sequences"]["flow"]    # [MW] -> power bought on DAA per interval
        sell_daa_pwr = results[(b_el, sink_daa)]["sequences"]["flow"]     # [MW] -> power sold on DAA per interval
        if self.pfs_daa and self.run_ida:
            buy_ida_pwr_daa = results[(source_ida, b_el)]["sequences"]["flow"]
            sell_ida_pwr_daa = results[(b_el, sink_ida)]["sequences"]["flow"]
        else:
            buy_ida_pwr_daa = zero_series.copy()
            sell_ida_pwr_daa = zero_series.copy()
        if self.pfs_daa and self.run_idc:
            buy_idc_pwr_daa = results[(source_idc, b_el)]["sequences"]["flow"]
            sell_idc_pwr_daa = results[(b_el, sink_idc)]["sequences"]["flow"]
        else:
            buy_idc_pwr_daa = zero_series.copy()
            sell_idc_pwr_daa = zero_series.copy()

        # Transform power-series to energy-series (pd.Series -> pd.Series)
        cha_daa = cha_daa_pwr * dt_h                    # [MWh] -> charged energy per interval
        discha_daa = discha_daa_pwr * dt_h              # [MWh] -> discharged energy per interval
        soc_daa_horizon = seq_storage["storage_content"]# [MWh]
        buy_daa = buy_daa_pwr * dt_h                    # [MWh]
        sell_daa = sell_daa_pwr * dt_h                  # [MWh]
        buy_ida_daa = buy_ida_pwr_daa * dt_h            # [MWh] 
        sell_ida_daa = sell_ida_pwr_daa * dt_h          # [MWh]
        buy_idc_daa = buy_idc_pwr_daa * dt_h            # [MWh]
        sell_idc_daa = sell_idc_pwr_daa * dt_h          # [MWh]
        

        #--- 6) Cost calculation over optimization horizon ---

        profit_daa_horizon_series = ((sell_daa - buy_daa) * price_daa.reindex(timeindex))   # Profit Series               
        profit_daa_horizon = float(profit_daa_horizon_series.sum())                         # Netto profit over full optimization horizon
        profit_ida_horizon = float(((sell_ida_daa - buy_ida_daa) * price_ida.reindex(timeindex)).sum())
        profit_idc_horizon = float(((sell_idc_daa - buy_idc_daa) * price_idc.reindex(timeindex)).sum())


        #--- 7) Rolling Horizon commit ---

        commit_end_eps = self._commit_end_eps(timeindex)
        profit_daa_commit_series = ((sell_daa - buy_daa) * price_daa.reindex(timeindex)).loc[:commit_end_eps]
        profit_daa_commit = float(profit_daa_commit_series.sum())
        profit_ida_commit = float((((sell_ida_daa - buy_ida_daa) * price_ida.reindex(timeindex)).loc[:commit_end_eps]).sum())
        profit_idc_commit = float((((sell_idc_daa - buy_idc_daa) * price_idc.reindex(timeindex)).loc[:commit_end_eps]).sum())
        soc_daa_commit = float(soc_daa_horizon.loc[:commit_end_eps].iloc[-1])           # SoC at the end of the commit period -> loc cuts the pandas series, iloc[-1] takes the last value of the cut series


        #--- 8) Calculate market dispatch ---

        market_dispatch_daa_pwr = calculate_market_dispatch.build_market_dispatch_series(buy_daa_pwr, sell_daa_pwr)
        market_dispatch_daa = market_dispatch_daa_pwr * dt_h
        start_soc_daa = self.initial_soc * self.capacity
        soc_market_dispatch_daa = calculate_market_dispatch.build_market_dispatch_soc(market_dispatch_daa_pwr, start_soc_daa, dt_h, self.eff_in, self.eff_out)
        soc_market_dispatch_daa_commit = float(soc_market_dispatch_daa.loc[:commit_end_eps].iloc[-1])


        #--- 9) Log results ---
        
        self.log.info(f"  Net profit DAA after {self.commit_hours} h : {profit_daa_commit:.2f} EUR")
        self.log.info(f"  Net profit DAA after {self.lookahead_hours} h : {profit_daa_horizon:.2f} EUR")
        


        return {
            "cha_daa": cha_daa,
            "cha_daa_pwr": cha_daa_pwr,
            "discha_daa": discha_daa,
            "discha_daa_pwr": discha_daa_pwr,
            "buy_daa": buy_daa,
            "buy_daa_pwr": buy_daa_pwr,
            "sell_daa": sell_daa,
            "sell_daa_pwr": sell_daa_pwr,
            "buy_ida_daa": buy_ida_daa,
            "buy_ida_pwr_daa": buy_ida_pwr_daa,
            "sell_ida_daa": sell_ida_daa,
            "sell_ida_pwr_daa": sell_ida_pwr_daa,
            "buy_idc_daa": buy_idc_daa,
            "buy_idc_pwr_daa": buy_idc_pwr_daa,
            "sell_idc_daa": sell_idc_daa,
            "sell_idc_pwr_daa": sell_idc_pwr_daa,
            "soc_daa_horizon": soc_daa_horizon,
            "soc_daa_commit": soc_daa_commit,
            "market_dispatch_daa_pwr": market_dispatch_daa_pwr,
            "market_dispatch_daa": market_dispatch_daa,
            "soc_market_dispatch_daa": soc_market_dispatch_daa,
            "soc_market_dispatch_daa_commit": soc_market_dispatch_daa_commit,
            "profit_daa_horizon": profit_daa_horizon,
            "profit_daa_horizon_series": profit_daa_horizon_series,
            "profit_daa_commit": profit_daa_commit,
            "profit_daa_commit_series": profit_daa_commit_series,
            "profit_ida_horizon": profit_ida_horizon,
            "profit_ida_commit": profit_ida_commit,
            "profit_idc_horizon": profit_idc_horizon,
            "profit_idc_commit": profit_idc_commit,
            "max_cha_pwr_fca": self.max_cha_pwr_fca,
            "max_discha_pwr_fca": self.max_discha_pwr_fca,
            "max_cha_pwr_active": max_cha_pwr_active,
            "max_discha_pwr_active": max_discha_pwr_active,
        }
    
    def _no_dispatch_daa(self):  # Helper function to create empty dispatch for DAA
        
        idx = self.df_DAA.index
        empty_series = pd.Series(0.0, index=idx)
        return {
            "cha_daa": empty_series,
            "cha_daa_pwr": empty_series,
            "discha_daa": empty_series,
            "discha_daa_pwr": empty_series,
            "buy_daa": empty_series,
            "buy_daa_pwr": empty_series,
            "sell_daa": empty_series,
            "sell_daa_pwr": empty_series,
            "buy_ida_daa": empty_series,
            "buy_ida_pwr_daa": empty_series,
            "sell_ida_daa": empty_series,
            "sell_ida_pwr_daa": empty_series,
            "buy_idc_daa": empty_series,
            "buy_idc_pwr_daa": empty_series,
            "sell_idc_daa": empty_series,
            "sell_idc_pwr_daa": empty_series,
            "soc_daa_horizon": pd.Series(self.initial_soc * self.capacity, index=idx),  # SoC remains constant at initial level
            "soc_daa_commit": self.initial_soc * self.capacity,
            "market_dispatch_daa_pwr": empty_series,
            "market_dispatch_daa": empty_series,
            "soc_market_dispatch_daa": pd.Series(self.initial_soc * self.capacity, index=idx),
            "soc_market_dispatch_daa_commit": self.initial_soc * self.capacity,
            "profit_daa_horizon": 0.0,
            "profit_daa_commit": 0.0,
            "profit_daa_horizon_series": empty_series,
            "profit_daa_commit_series": empty_series,
            "profit_ida_horizon": 0.0,
            "profit_ida_commit": 0.0,
            "profit_idc_horizon": 0.0,
            "profit_idc_commit": 0.0,
            "max_cha_pwr_fca": self.max_cha_pwr_fca,
            "max_discha_pwr_fca": self.max_discha_pwr_fca,
            "max_cha_pwr_active": self.max_cha_pwr_fca,
            "max_discha_pwr_active": self.max_discha_pwr_fca,
        }
    


    # ------------------------------------------------------------
    # Intra-Day-Auction Optimization (Class Method)
    # ------------------------------------------------------------

    def optimize_ida(self, buy_daa_pwr: pd.Series, sell_daa_pwr: pd.Series, min_soc_for_fcr: pd.Series, max_soc_for_fcr: pd.Series, marketed_pwr_fcr: pd.Series):

        log = self.log
        log.info("Building IDA optimization system...")

        # --- 0) Time index and data alignment ---

        df = self.df_all_markets.copy()

        timeindex = df.index
        dt_h = (df.index[1] - df.index[0]).total_seconds() / 3600.0
        if abs(dt_h - 0.25) > 1e-9:
            raise ValueError(f"optimize_ida expects a 15min master index, got dt_h={dt_h} h.")

        horizon_hours = len(timeindex) * dt_h
        t_end_commit = timeindex[0] + pd.Timedelta(hours=self.commit_hours)
        t_end_horizon = timeindex[0] + pd.Timedelta(hours=horizon_hours)
        T_to_commit = [t for t, ts in enumerate(timeindex) if ts < t_end_commit]
        T_to_horizon_without_commit = [t for t, ts in enumerate(timeindex) if (ts >= t_end_commit and ts < t_end_horizon)]

        price_ida = df["price_ida"].astype(float)
        price_idc = df["price_idc"].astype(float)

        buy_daa_pwr_15 = transform_series.reindex_to_target_index(buy_daa_pwr, timeindex, fill_value=0.0)
        fixed_daa_buy = buy_daa_pwr_15 * dt_h # [MWh]
        sell_daa_pwr_15 = transform_series.reindex_to_target_index(sell_daa_pwr, timeindex, fill_value=0.0)
        fixed_daa_sell = sell_daa_pwr_15 * dt_h # [MWh]


        # --- 1) Build energy system ---

        es = solph.EnergySystem(timeindex=timeindex, infer_last_interval=True)
        b_el = solph.Bus(label="electricity")

        # Fixed DAA buy/sell from previous stage
        daa_in_fix, daa_in_nom = transform_series.series_to_fix_and_nominal(buy_daa_pwr_15)
        source_daa = solph.components.Source(
            label="source_daa",
            outputs={
                b_el: solph.Flow(
                    fix=daa_in_fix,
                    nominal_value=daa_in_nom,
                    variable_costs=0.0,
                )
            } if daa_in_fix is not None else {}
        )
        daa_out_fix, daa_out_nom = transform_series.series_to_fix_and_nominal(sell_daa_pwr_15)
        sink_daa = solph.components.Sink(
            label="sink_daa",
            inputs={
                b_el: solph.Flow(
                    fix=daa_out_fix,
                    nominal_value=daa_out_nom,
                    variable_costs=0.0,
                )
            } if daa_out_fix is not None else {}
        )

        # Variable IDA market
        source_ida = solph.components.Source(
            label="source_ida",
            outputs={
                b_el: solph.Flow(
                    nominal_value=self.max_charge_power,
                    variable_costs=(price_ida + self.transaction_costs).values,
                )
            },
        )
        sink_ida = solph.components.Sink(
            label="sink_ida",
            inputs={
                b_el: solph.Flow(
                    nominal_value=self.max_discharge_power,
                    variable_costs=-(price_ida - self.transaction_costs).values,
                )
            },
        )
        
        # Variable IDC forecast (perfect foresight mode)
        source_idc = solph.components.Source(
            label="source_idc",
            outputs={
                b_el: solph.Flow(
                    nominal_value=self.max_charge_power,
                    variable_costs=(price_idc + self.transaction_costs).values,
                )
            } if self.pfs_ida and self.run_idc else {}
        )
        sink_idc = solph.components.Sink(
            label="sink_idc",
            inputs={
                b_el: solph.Flow(
                    nominal_value=self.max_discharge_power,
                    variable_costs=-(price_idc - self.transaction_costs).values,
                )
            } if self.pfs_ida and self.run_idc else {}
        )

        # Battery storage:
        # Defining SoC limits by FCR and Power limits by FCR and FCA (data allignment: pd.Series to sequences as required by oemof)
        min_level_seq, max_level_seq = transform_series.make_storage_level_sequences(
            timeindex=timeindex,
            min_soc_mwh=min_soc_for_fcr,
            max_soc_mwh=max_soc_for_fcr,
            capacity_mwh=self.capacity,
        )
        # Compute active power limits: FCA ceiling minus FCR reservation
        active_limits = get_fca_constraints.compute_active_power_limits(
            max_cha_pwr_fca=self.max_cha_pwr_fca,
            max_discha_pwr_fca=self.max_discha_pwr_fca,
            marketed_pwr_fcr=marketed_pwr_fcr,
        )
        max_cha_pwr_active = active_limits["max_cha_pwr_active"]
        max_discha_pwr_active = active_limits["max_discha_pwr_active"]
        max_in_seq = transform_series.make_flow_max_sequences(
            timeindex=timeindex,
            max_power_mw=max_cha_pwr_active,
            nominal_power_mw=self.max_charge_power,
        )
        max_out_seq = transform_series.make_flow_max_sequences(
            timeindex=timeindex,
            max_power_mw=max_discha_pwr_active,
            nominal_power_mw=self.max_discharge_power,
        )
        # Defining the batttery
        battery = solph.components.GenericStorage(
            label="battery",
            nominal_storage_capacity=self.capacity,
            inputs={b_el: solph.Flow(nominal_value=self.max_charge_power, max=max_in_seq)},
            outputs={b_el: solph.Flow(nominal_value=self.max_discharge_power, max=max_out_seq)},
            loss_rate=0.0,
            initial_storage_level=self.initial_soc,
            min_storage_level=min_level_seq,
            max_storage_level=max_level_seq,
            balanced=False,
            inflow_conversion_factor=self.eff_in,
            outflow_conversion_factor=self.eff_out,
        )

        # Adding components 
        es.add(b_el, source_ida, sink_ida, battery)
        if daa_in_fix is not None:
            es.add(source_daa)
        if daa_out_fix is not None:
            es.add(sink_daa)
        if self.pfs_ida and self.run_idc:
            es.add(source_idc, sink_idc)


        # --- 2) Build model ---

        log.info("Creating IDA model...")
        model = solph.Model(es)
        T = list(model.TIMESTEPS)


        # --- 3) Add custom constraints ---

        # (1) Cycle Limit
        blk_cycles = po.Block()
        model.add_component("ida_cycle_limit_block", blk_cycles)
        max_total_charge_per_day = self.max_cycles * self.capacity
        lookahead_hours_without_commit = max(0.0, horizon_hours - self.commit_hours)
        max_total_charge_in_lookahead = self.max_cycles * self.capacity * (lookahead_hours_without_commit / 24.0)
        def _max_cycles_day_rule(b):
            return sum(model.flow[b_el, battery, t] * dt_h for t in T_to_commit) <= max_total_charge_per_day
        def _max_cycles_lookahead_rule(b):
            if lookahead_hours_without_commit <= 0 or len(T_to_horizon_without_commit) == 0:
                return po.Constraint.Skip
            return sum(model.flow[b_el, battery, t] * dt_h for t in T_to_horizon_without_commit) <= max_total_charge_in_lookahead
        blk_cycles.max_cycles_per_day = po.Constraint(rule=_max_cycles_day_rule)
        blk_cycles.max_cycles_lookahead = po.Constraint(rule=_max_cycles_lookahead_rule)

        # (2) No simultaneous buy/sell in IDA
        blk_ida_mode = po.Block()
        model.add_component("ida_no_simul_block", blk_ida_mode)
        blk_ida_mode.y_buy = po.Var(T, domain=po.Binary)
        def _ida_buy_limit_rule(b, t):
            return model.flow[source_ida, b_el, t] <= self.max_charge_power * b.y_buy[t]
        def _ida_sell_limit_rule(b, t):
            return model.flow[b_el, sink_ida, t] <= self.max_discharge_power * (1 - b.y_buy[t])
        blk_ida_mode.buy_limit = po.Constraint(T, rule=_ida_buy_limit_rule)
        blk_ida_mode.sell_limit = po.Constraint(T, rule=_ida_sell_limit_rule)

        # (3) No simultaneous buy/sell in IDC if perfect foresight is active
        if self.pfs_ida and self.run_idc:
            blk_idc_mode = po.Block()
            model.add_component("idc_no_simul_block", blk_idc_mode)
            blk_idc_mode.y_buy = po.Var(T, domain=po.Binary)
            def _idc_buy_limit_rule(b, t):
                return model.flow[source_idc, b_el, t] <= self.max_charge_power * b.y_buy[t]
            def _idc_sell_limit_rule(b, t):
                return model.flow[b_el, sink_idc, t] <= self.max_discharge_power * (1 - b.y_buy[t])
            blk_idc_mode.buy_limit = po.Constraint(T, rule=_idc_buy_limit_rule)
            blk_idc_mode.sell_limit = po.Constraint(T, rule=_idc_sell_limit_rule)


        # --- 4) Solve ---

        log.info("Solving IDA with %s solver...", self.solver)
        model.solve(solver=self.solver, solve_kwargs={"tee": False})


        # --- 5) Extract results ---

        log.info("Processing IDA results...")
        results = solph.processing.results(model)
        zero_series = pd.Series(0.0, index=timeindex)

        # Flow Results [MW]
        cha_ida_pwr = results[(b_el, battery)]["sequences"]["flow"]
        discha_ida_pwr = results[(battery, b_el)]["sequences"]["flow"]
        soc_ida_horizon = results[(battery, None)]["sequences"]["storage_content"]
        buy_ida_pwr = results[(source_ida, b_el)]["sequences"]["flow"]
        sell_ida_pwr = results[(b_el, sink_ida)]["sequences"]["flow"]

        if self.pfs_ida and self.run_idc:
            buy_idc_pwr_ida = results[(source_idc, b_el)]["sequences"]["flow"]
            sell_idc_pwr_ida = results[(b_el, sink_idc)]["sequences"]["flow"]
        else:
            buy_idc_pwr_ida = zero_series.copy()
            sell_idc_pwr_ida = zero_series.copy()

        # Energy Results [MWh]
        cha_ida = cha_ida_pwr * dt_h
        discha_ida = discha_ida_pwr * dt_h
        buy_ida = buy_ida_pwr * dt_h
        sell_ida = sell_ida_pwr * dt_h
        buy_idc_ida = buy_idc_pwr_ida * dt_h
        sell_idc_ida = sell_idc_pwr_ida * dt_h


        # --- 6) Profit calculation over optimization horizon ---

        profit_ida_horizon_series = ((sell_ida - buy_ida) * price_ida.reindex(timeindex))   # Profit Series
        profit_ida_horizon = float(profit_ida_horizon_series.sum())
        profit_idc_horizon = float(((sell_idc_ida - buy_idc_ida) * price_idc.reindex(timeindex)).sum())


        # --- 7) Rolling horizon commit ---

        commit_end_eps = self._commit_end_eps(timeindex)
        profit_series_ida = (sell_ida.fillna(0.0) - buy_ida.fillna(0.0)) * price_ida.reindex(timeindex)
        profit_ida_commit_series = profit_series_ida.loc[:commit_end_eps]
        profit_ida_commit = float(profit_ida_commit_series.sum())
        profit_idc_commit = float((((sell_idc_ida - buy_idc_ida) * price_idc.reindex(timeindex)).loc[:commit_end_eps]).sum())
        soc_ida_commit = float(soc_ida_horizon.loc[:commit_end_eps].iloc[-1])


        # --- 8) Calculate market dispatch (cumulated DAA + IDA: buy_daa_15 + buy_ida - sell_daa_15 - sell_ida) ---

        market_dispatch_ida_pwr = calculate_market_dispatch.build_market_dispatch_series(buy_daa_pwr_15 + buy_ida_pwr, sell_daa_pwr_15 + sell_ida_pwr)
        market_dispatch_ida = market_dispatch_ida_pwr * dt_h
        start_soc_ida = self.initial_soc * self.capacity
        soc_market_dispatch_ida = calculate_market_dispatch.build_market_dispatch_soc(market_dispatch_ida_pwr, start_soc_ida, dt_h, self.eff_in, self.eff_out)
        soc_market_dispatch_ida_commit = float(soc_market_dispatch_ida.loc[:commit_end_eps].iloc[-1])


        # --- 9) Log results ---

        log.info("sum fixed daa buy-position = %.6f", float(buy_daa_pwr_15.sum()))
        log.info("sum fixed daa sell-position = %.6f", float(sell_daa_pwr_15.sum()))
        log.info("sum buy_ida_pwr = %.6f", float(buy_ida_pwr.sum()))
        log.info("sum sell_ida_pwr = %.6f", float(sell_ida_pwr.sum()))
        log.info("sum buy_idc_pwr_ida = %.6f", float(buy_idc_pwr_ida.sum()))
        log.info("sum sell_idc_pwr_ida = %.6f", float(sell_idc_pwr_ida.sum()))
        log.info("soc_ida_horizon = %s", soc_ida_horizon)
        log.info("soc_ida_commit = %s", soc_ida_commit)
        log.info("IDA profit horizon: %.2f EUR", profit_ida_horizon)
        log.info("IDA profit commit: %.2f EUR", profit_ida_commit)
        log.info("IDC profit horizon: %.2f EUR", profit_idc_horizon)
        log.info("IDC profit commit: %.2f EUR", profit_idc_commit)

        return {
            "cha_ida": cha_ida,
            "cha_ida_pwr": cha_ida_pwr,
            "discha_ida": discha_ida,
            "discha_ida_pwr": discha_ida_pwr,
            "buy_ida": buy_ida,
            "buy_ida_pwr": buy_ida_pwr,
            "sell_ida": sell_ida,
            "sell_ida_pwr": sell_ida_pwr,
            "buy_idc_ida": buy_idc_ida,
            "buy_idc_pwr_ida": buy_idc_pwr_ida,
            "sell_idc_ida": sell_idc_ida,
            "sell_idc_pwr_ida": sell_idc_pwr_ida,
            "soc_ida_horizon": soc_ida_horizon,
            "soc_ida_commit": soc_ida_commit,
            "market_dispatch_ida_pwr": market_dispatch_ida_pwr,
            "market_dispatch_ida": market_dispatch_ida,
            "soc_market_dispatch_ida": soc_market_dispatch_ida,
            "soc_market_dispatch_ida_commit": soc_market_dispatch_ida_commit,
            "profit_ida_horizon": profit_ida_horizon,
            "profit_ida_horizon_series": profit_ida_horizon_series,
            "profit_ida_commit": profit_ida_commit,
            "profit_ida_commit_series": profit_ida_commit_series,
            "profit_idc_horizon": profit_idc_horizon,
            "profit_idc_commit": profit_idc_commit,
            "fixed_daa_buy": fixed_daa_buy,
            "fixed_daa_sell": fixed_daa_sell,
            "max_cha_pwr_fca": self.max_cha_pwr_fca,
            "max_discha_pwr_fca": self.max_discha_pwr_fca,
            "max_cha_pwr_active": max_cha_pwr_active,
            "max_discha_pwr_active": max_discha_pwr_active,
        }
    
    def _no_dispatch_ida(self, buy_daa_pwr: pd.Series, sell_daa_pwr: pd.Series):  # Helper function to create empty dispatch for IDA
        idx = self.df_IDA.index
        dt_h = 0.25  # 15-min resolution
        empty_series = pd.Series(0.0, index=idx)
        buy_daa_pwr_15 = transform_series.reindex_to_target_index(buy_daa_pwr, idx, fill_value=0.0)
        fixed_daa_buy = buy_daa_pwr_15 * dt_h
        sell_daa_pwr_15 = transform_series.reindex_to_target_index(sell_daa_pwr, idx, fill_value=0.0)
        fixed_daa_sell = sell_daa_pwr_15 * dt_h
        return {
            "cha_ida": empty_series,
            "cha_ida_pwr": empty_series,
            "discha_ida": empty_series,
            "discha_ida_pwr": empty_series,
            "buy_ida": empty_series,
            "buy_ida_pwr": empty_series,
            "sell_ida": empty_series,
            "sell_ida_pwr": empty_series,
            "buy_idc_ida": empty_series,
            "buy_idc_pwr_ida": empty_series,
            "sell_idc_ida": empty_series,
            "sell_idc_pwr_ida": empty_series,
            "soc_ida_horizon": pd.Series(self.initial_soc * self.capacity, index=idx),  # SoC remains constant at initial level
            "soc_ida_commit": self.initial_soc * self.capacity,
            "market_dispatch_ida_pwr": empty_series,
            "market_dispatch_ida": empty_series,
            "soc_market_dispatch_ida": pd.Series(self.initial_soc * self.capacity, index=idx),
            "soc_market_dispatch_ida_commit": self.initial_soc * self.capacity,
            "profit_ida_horizon": 0.0,
            "profit_ida_horizon_series": empty_series,
            "profit_ida_commit": 0.0,
            "profit_ida_commit_series": empty_series,
            "profit_idc_horizon": 0.0,
            "profit_idc_commit": 0.0,
            "fixed_daa_buy": fixed_daa_buy,
            "fixed_daa_sell": fixed_daa_sell,
            "max_cha_pwr_fca": self.max_cha_pwr_fca,
            "max_discha_pwr_fca": self.max_discha_pwr_fca,
            "max_cha_pwr_active": self.max_cha_pwr_fca,
            "max_discha_pwr_active": self.max_discha_pwr_fca,
        }



    # ------------------------------------------------------------
    # Intra-Day-Continious Optimization (Class Method)
    # ------------------------------------------------------------

    def optimize_idc(self, buy_daa_pwr: pd.Series, sell_daa_pwr: pd.Series, buy_ida_pwr: pd.Series, sell_ida_pwr: pd.Series, min_soc_for_fcr: pd.Series, max_soc_for_fcr: pd.Series, marketed_pwr_fcr: pd.Series):

        log = self.log
        log.info("Building IDC optimization system...")


        # --- 0) Time index and data alignment ---

        df = self.df_all_markets.copy()
        timeindex = df.index
        dt_h = (df.index[1] - df.index[0]).total_seconds() / 3600.0
        if abs(dt_h - 0.25) > 1e-9:
            raise ValueError(f"optimize_idc expects a 15min master index, got dt_h={dt_h} h.")

        horizon_hours = len(timeindex) * dt_h
        t_end_commit = timeindex[0] + pd.Timedelta(hours=self.commit_hours)
        t_end_horizon = timeindex[0] + pd.Timedelta(hours=horizon_hours)
        T_to_commit = [t for t, ts in enumerate(timeindex) if ts < t_end_commit]
        T_to_horizon_without_commit = [t for t, ts in enumerate(timeindex) if (ts >= t_end_commit and ts < t_end_horizon)]

        price_idc = df["price_idc"].astype(float)

        buy_daa_pwr_15 = transform_series.reindex_to_target_index(buy_daa_pwr, timeindex, fill_value=0.0)
        fixed_daa_buy = buy_daa_pwr_15 * dt_h # [MWh]
        sell_daa_pwr_15 = transform_series.reindex_to_target_index(sell_daa_pwr, timeindex, fill_value=0.0)
        fixed_daa_sell = sell_daa_pwr_15 * dt_h # [MWh]
        buy_ida_pwr_15 = transform_series.reindex_to_target_index(buy_ida_pwr, timeindex, fill_value=0.0)
        fixed_ida_buy = buy_ida_pwr_15 * dt_h # [MWh]
        sell_ida_pwr_15 = transform_series.reindex_to_target_index(sell_ida_pwr, timeindex, fill_value=0.0)
        fixed_ida_sell = sell_ida_pwr_15 * dt_h # [MWh]

        
        # --- 1) Build energy system ---

        es = solph.EnergySystem(timeindex=timeindex, infer_last_interval=True)
        b_el = solph.Bus(label="electricity")

        # Fixed DAA positions
        daa_in_fix, daa_in_nom = transform_series.series_to_fix_and_nominal(buy_daa_pwr_15)
        source_daa = solph.components.Source(
            label="source_daa",
            outputs={
                b_el: solph.Flow(
                    fix=daa_in_fix,
                    nominal_value=daa_in_nom,
                    variable_costs=0.0,
                )
            } if daa_in_fix is not None else {}
        )
        daa_out_fix, daa_out_nom = transform_series.series_to_fix_and_nominal(sell_daa_pwr_15)
        sink_daa = solph.components.Sink(
            label="sink_daa",
            inputs={
                b_el: solph.Flow(
                    fix=daa_out_fix,
                    nominal_value=daa_out_nom,
                    variable_costs=0.0,
                )
            } if daa_out_fix is not None else {}
        )

        # Fixed IDA positions
        ida_in_fix, ida_in_nom = transform_series.series_to_fix_and_nominal(buy_ida_pwr_15)
        source_ida = solph.components.Source(
            label="source_ida",
            outputs={
                b_el: solph.Flow(
                    fix=ida_in_fix,
                    nominal_value=ida_in_nom,
                    variable_costs=0.0,
                )
            } if ida_in_fix is not None else {}
        )
        ida_out_fix, ida_out_nom = transform_series.series_to_fix_and_nominal(sell_ida_pwr_15)
        sink_ida = solph.components.Sink(
            label="sink_ida",
            inputs={
                b_el: solph.Flow(
                    fix=ida_out_fix,
                    nominal_value=ida_out_nom,
                    variable_costs=0.0,
                )
            } if ida_out_fix is not None else {}
        )

        # Variable IDC buy / sell
        source_idc = solph.components.Source(
            label="source_idc",
            outputs={
                b_el: solph.Flow(
                    nominal_value=self.max_charge_power,
                    variable_costs=(price_idc + self.transaction_costs).values,
                )
            },
        )
        sink_idc = solph.components.Sink(
            label="sink_idc",
            inputs={
                b_el: solph.Flow(
                    nominal_value=self.max_discharge_power,
                    variable_costs=-(price_idc - self.transaction_costs).values,
                )
            },
        )

        # Battery storage:
        # Defining SoC limits by FCR and Power limits by FCR and FCA (data allignment: pd.Series to sequences as required by oemof)
        min_level_seq, max_level_seq = transform_series.make_storage_level_sequences(
            timeindex=timeindex,
            min_soc_mwh=min_soc_for_fcr,
            max_soc_mwh=max_soc_for_fcr,
            capacity_mwh=self.capacity,
        )
        # Compute active power limits: FCA ceiling minus FCR reservation
        active_limits = get_fca_constraints.compute_active_power_limits(
            max_cha_pwr_fca=self.max_cha_pwr_fca,
            max_discha_pwr_fca=self.max_discha_pwr_fca,
            marketed_pwr_fcr=marketed_pwr_fcr,
        )
        max_cha_pwr_active = active_limits["max_cha_pwr_active"]
        max_discha_pwr_active = active_limits["max_discha_pwr_active"]
        max_in_seq = transform_series.make_flow_max_sequences(
            timeindex=timeindex,
            max_power_mw=max_cha_pwr_active,
            nominal_power_mw=self.max_charge_power,
        )
        max_out_seq = transform_series.make_flow_max_sequences(
            timeindex=timeindex,
            max_power_mw=max_discha_pwr_active,
            nominal_power_mw=self.max_discharge_power,
        )
        # Defining the battery
        battery = solph.components.GenericStorage(
            label="battery",
            nominal_storage_capacity=self.capacity,
            inputs={b_el: solph.Flow(nominal_value=self.max_charge_power, max=max_in_seq)},
            outputs={b_el: solph.Flow(nominal_value=self.max_discharge_power, max=max_out_seq)},
            loss_rate=0.0,
            initial_storage_level=self.initial_soc,
            min_storage_level=min_level_seq,
            max_storage_level=max_level_seq,
            balanced=False,
            inflow_conversion_factor=self.eff_in,
            outflow_conversion_factor=self.eff_out,
        )

        es.add(b_el, source_idc, sink_idc, battery)
        if daa_in_fix is not None: es.add(source_daa)
        if daa_out_fix is not None: es.add(sink_daa)
        if ida_in_fix is not None: es.add(source_ida)
        if ida_out_fix is not None: es.add(sink_ida)


        # --- 2) Create model ---

        log.info("Creating IDC model...")
        model = solph.Model(es)
        T = list(model.TIMESTEPS)


        # --- 3) Constraints ---

        # (1) Cycle limit
        blk_cycles = po.Block()
        model.add_component("idc_cycle_limit_block", blk_cycles)
        max_total_charge_per_day = self.max_cycles * self.capacity
        lookahead_hours_without_commit = max(0.0, horizon_hours - self.commit_hours)
        max_total_charge_in_lookahead = self.max_cycles * self.capacity * (lookahead_hours_without_commit / 24.0)
        def _max_cycles_day_rule(b):
            return sum(model.flow[b_el, battery, t] * dt_h for t in T_to_commit) <= max_total_charge_per_day
        def _max_cycles_lookahead_rule(b):
            if lookahead_hours_without_commit <= 0 or len(T_to_horizon_without_commit) == 0:
                return po.Constraint.Skip
            return sum(model.flow[b_el, battery, t] * dt_h for t in T_to_horizon_without_commit) <= max_total_charge_in_lookahead
        blk_cycles.max_cycles_per_day = po.Constraint(rule=_max_cycles_day_rule)
        blk_cycles.max_cycles_lookahead = po.Constraint(rule=_max_cycles_lookahead_rule)
        
        # (2) No simultaneous buy/sell in IDC
        blk_idc_mode = po.Block()
        model.add_component("idc_no_simul_block", blk_idc_mode)
        blk_idc_mode.y_buy = po.Var(T, domain=po.Binary)
        def _idc_buy_limit_rule(b, t):
            return model.flow[source_idc, b_el, t] <= self.max_charge_power * b.y_buy[t]
        def _idc_sell_limit_rule(b, t):
            return model.flow[b_el, sink_idc, t] <= self.max_discharge_power * (1 - b.y_buy[t])
        blk_idc_mode.buy_limit = po.Constraint(T, rule=_idc_buy_limit_rule)
        blk_idc_mode.sell_limit = po.Constraint(T, rule=_idc_sell_limit_rule)


        # --- 4) Solve ---

        log.info("Solving IDC with %s solver...", self.solver)
        model.solve(solver=self.solver, solve_kwargs={"tee": False})


        # --- 5) Extract results ---

        log.info("Processing IDC results...")
        results = solph.processing.results(model)

        # Flow results [MW]
        cha_idc_pwr = results[(b_el, battery)]["sequences"]["flow"]
        discha_idc_pwr = results[(battery, b_el)]["sequences"]["flow"]
        soc_idc_horizon = results[(battery, None)]["sequences"]["storage_content"]
        buy_idc_pwr = results[(source_idc, b_el)]["sequences"]["flow"]
        sell_idc_pwr = results[(b_el, sink_idc)]["sequences"]["flow"]

        # Energy Results [MWh]
        cha_idc = cha_idc_pwr * dt_h
        discha_idc = discha_idc_pwr * dt_h
        buy_idc = buy_idc_pwr * dt_h
        sell_idc = sell_idc_pwr * dt_h


        # --- 6) Profit calculation ---
        
        profit_idc_horizon_series = ((sell_idc - buy_idc) * price_idc.reindex(timeindex))   # Profit Series
        profit_idc_horizon = float(profit_idc_horizon_series.sum())


        # --- 7) Rolling Horizon Commit ---

        commit_end_eps = self._commit_end_eps(timeindex)
        profit_idc_commit_series = ((sell_idc.fillna(0.0) - buy_idc.fillna(0.0)) * price_idc.reindex(timeindex)).loc[:commit_end_eps]
        profit_idc_commit = float(profit_idc_commit_series.sum())
        soc_idc_commit = float(soc_idc_horizon.loc[:commit_end_eps].iloc[-1])


        # --- 8) Calculate market dispatch series (cumulated DAA + IDA + IDC) ---

        market_dispatch_idc_pwr = calculate_market_dispatch.build_market_dispatch_series(
            buy_daa_pwr_15 + buy_ida_pwr_15 + buy_idc_pwr,
            sell_daa_pwr_15 + sell_ida_pwr_15 + sell_idc_pwr,
        )
        market_dispatch_idc = market_dispatch_idc_pwr * dt_h
        start_soc_idc = self.initial_soc * self.capacity
        soc_market_dispatch_idc = calculate_market_dispatch.build_market_dispatch_soc(market_dispatch_idc_pwr, start_soc_idc, dt_h, self.eff_in, self.eff_out)
        soc_market_dispatch_idc_commit = float(soc_market_dispatch_idc.loc[:commit_end_eps].iloc[-1])


        # --- 9) Log results ---
        log.info("------------------------------------------------")
        log.info("fixed daa buy-position = %.6f", float(buy_daa_pwr_15.sum()))
        log.info("fixed daa sell-position = %.6f", float(sell_daa_pwr_15.sum()))
        log.info("fixed ida buy-position = %.6f", float(buy_ida_pwr_15.sum()))
        log.info("fixed ida sell-position = %.6f", float(sell_ida_pwr_15.sum()))
        log.info("sum buy_idc_pwr = %.6f", float(buy_idc_pwr.sum()))
        log.info("sum sell_idc_pwr = %.6f", float(sell_idc_pwr.sum()))
        log.info("soc_idc_horizon = %s", soc_idc_horizon)
        log.info("soc_idc_commit = %s", soc_idc_commit)
        log.info("IDC profit horizon: %.2f EUR", profit_idc_horizon)
        log.info("IDC profit commit: %.2f EUR", profit_idc_commit)
        log.info("------------------------------------------------")

        return {
            "cha_idc": cha_idc,
            "cha_idc_pwr": cha_idc_pwr,
            "discha_idc": discha_idc,
            "discha_idc_pwr": discha_idc_pwr,
            "buy_idc": buy_idc,
            "buy_idc_pwr": buy_idc_pwr,
            "sell_idc": sell_idc,
            "sell_idc_pwr": sell_idc_pwr,
            "soc_idc_horizon": soc_idc_horizon,
            "soc_idc_commit": soc_idc_commit,
            "market_dispatch_idc_pwr": market_dispatch_idc_pwr,
            "market_dispatch_idc": market_dispatch_idc,
            "soc_market_dispatch_idc": soc_market_dispatch_idc,
            "soc_market_dispatch_idc_commit": soc_market_dispatch_idc_commit,
            "profit_idc_horizon": profit_idc_horizon,
            "profit_idc_horizon_series": profit_idc_horizon_series,
            "profit_idc_commit": profit_idc_commit,
            "profit_idc_commit_series": profit_idc_commit_series,
            "fixed_daa_buy": fixed_daa_buy,
            "fixed_daa_sell": fixed_daa_sell,
            "fixed_ida_buy": fixed_ida_buy,
            "fixed_ida_sell": fixed_ida_sell,
            "max_cha_pwr_fca": self.max_cha_pwr_fca,
            "max_discha_pwr_fca": self.max_discha_pwr_fca,
            "max_cha_pwr_active": max_cha_pwr_active,
            "max_discha_pwr_active": max_discha_pwr_active,
        }
    
    def _no_dispatch_idc(self, buy_daa_pwr: pd.Series, sell_daa_pwr: pd.Series, buy_ida_pwr: pd.Series, sell_ida_pwr: pd.Series):  # Helper function to create empty dispatch for IDC
        idx = self.df_IDC.index
        dt_h = 0.25  # 15-min resolution
        empty_series = pd.Series(0.0, index=idx)
        buy_daa_pwr_15 = transform_series.reindex_to_target_index(buy_daa_pwr, idx, fill_value=0.0)
        fixed_daa_buy = buy_daa_pwr_15 * dt_h
        sell_daa_pwr_15 = transform_series.reindex_to_target_index(sell_daa_pwr, idx, fill_value=0.0)
        fixed_daa_sell = sell_daa_pwr_15 * dt_h
        buy_ida_pwr_15 = transform_series.reindex_to_target_index(buy_ida_pwr, idx, fill_value=0.0)
        fixed_ida_buy = buy_ida_pwr_15 * dt_h
        sell_ida_pwr_15 = transform_series.reindex_to_target_index(sell_ida_pwr, idx, fill_value=0.0)
        fixed_ida_sell = sell_ida_pwr_15 * dt_h
        return {
            "cha_idc": empty_series,
            "cha_idc_pwr": empty_series,
            "discha_idc": empty_series,
            "discha_idc_pwr": empty_series,
            "buy_idc": empty_series,
            "buy_idc_pwr": empty_series,
            "sell_idc": empty_series,
            "sell_idc_pwr": empty_series,
            "soc_idc_horizon": pd.Series(self.initial_soc * self.capacity, index=idx),  # SoC remains constant at initial level
            "soc_idc_commit": self.initial_soc * self.capacity,
            "market_dispatch_idc_pwr": empty_series,
            "soc_market_dispatch_idc": pd.Series(self.initial_soc * self.capacity, index=idx),
            "soc_market_dispatch_idc_commit": self.initial_soc * self.capacity,
            "profit_idc_horizon": 0.0,
            "profit_idc_commit": 0.0,
            "fixed_daa_buy": fixed_daa_buy,
            "fixed_daa_sell": fixed_daa_sell,
            "fixed_ida_buy": fixed_ida_buy,
            "fixed_ida_sell": fixed_ida_sell,
            "market_dispatch_idc": empty_series,
            "profit_idc_horizon_series": empty_series,
            "profit_idc_commit_series": empty_series,
            "max_cha_pwr_fca": self.max_cha_pwr_fca,
            "max_discha_pwr_fca": self.max_discha_pwr_fca,
            "max_cha_pwr_active": self.max_cha_pwr_fca,
            "max_discha_pwr_active": self.max_discha_pwr_fca,
        }

         


