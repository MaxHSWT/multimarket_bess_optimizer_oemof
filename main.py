# conda venv: conda activate oemof-solph-env

import logging
from omegaconf import DictConfig
import hydra
import pandas as pd
from pathlib import Path

# Main modules
from src.model.multi_market_optimizer import MultiMarketOptimizer
# Helper functions
from src.utils.import_and_load_data import load_all_markets_and_fca_horizon, check_data_availability, check_fca_availability
from src.utils.save_optimization_results import build_daily_row, init_results_csv, append_daily_result
from src.utils.transform_pd_dataframes import transform_all_markets_to_master_df
# Plot module
from src.visualization.plotter import MultiMarketPlotter


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):

    log = logging.getLogger(__name__)
    log.info("Starting Battery Optimizer")

    # Read necessary flags from config
    capacity = float(getattr(cfg.battery, "capacity", 1.0))
    max_power = float(getattr(cfg.battery, "max_charge_power", 1.0))
    initial_soc = float(getattr(cfg.battery, "initial_soc", 0.5))
    lookahead_hours = int(getattr(cfg.optimization.rolling_horizon, "lookahead_hours", 24))
    commit_hours = int(getattr(cfg.optimization.rolling_horizon, "commit_hours", 24))
    results_dir = str(cfg.plotting.output_dir)

    # Initializing Data Window and Result Holders
    tz = str(cfg.simulation.timezone)
    start_day = pd.Timestamp(cfg.simulation.start_date).tz_localize(tz).normalize()
    end_day = pd.Timestamp(cfg.simulation.end_date).tz_localize(tz).normalize()
    days = pd.date_range(start_day, end_day, freq="D", tz=tz)
    init_soc = initial_soc
    total_profit_for_observed_period = 0.0  # Sum over all committed days
    results_csv_path = Path(results_dir) / "results.csv"
    init_results_csv(results_csv_path)
    plotter = None  # initialized per day with fresh df_all_markets

    # Sanity checks
    if commit_hours != 24:
        raise ValueError("This implementation assumes commit_hours=24.")
    if lookahead_hours < commit_hours:
        raise ValueError("lookahead_hours must be >= commit_hours.")
    check_data_availability(cfg=cfg, start_day=start_day, end_day=end_day, lookahead_hours=lookahead_hours, markets=("FCR", "DAA", "IDA", "IDC"))
    check_fca_availability(cfg)
    log.info("Data availability check passed. Starting optimization loop over days.")


    # --------------------------------------------------------------
    # Executing Optimization over Data Window (including RH)
    # --------------------------------------------------------------

    for day in days:
        
        # --- Load data for current horizon ---
        horizon_start = day
        df_FCR, df_DAA, df_IDA, df_IDC, df_FCA = load_all_markets_and_fca_horizon(
            cfg, horizon_start, lookahead_hours,
            max_charge_power=float(cfg.battery.max_charge_power),
            max_discharge_power=float(cfg.battery.max_discharge_power),
        )

        # --- Initialize and run optimizer ---
        multi_market_optimizer = MultiMarketOptimizer(cfg, df_FCR, df_DAA, df_IDA, df_IDC, init_soc, df_FCA)
        multi_market_optimizer.run_full_day()
        daily_profit = multi_market_optimizer.daily_profit_commit
        end_of_day_soc = multi_market_optimizer.soc_commit

        # --- Setting up attributes for next day ---
        init_soc = end_of_day_soc / float(multi_market_optimizer.optimizer.capacity)
        total_profit_for_observed_period += daily_profit

        # --- Saving daily results to csv ---
        row = build_daily_row(
            day=day,
            daily_profit=daily_profit,
            end_of_day_soc=end_of_day_soc,
            total_profit_for_observed_period=total_profit_for_observed_period,
            result_fcr=multi_market_optimizer.result_fcr,
            result_daa=multi_market_optimizer.result_daa,
            result_ida=multi_market_optimizer.result_ida,
            result_idc=multi_market_optimizer.result_idc,
        )
        append_daily_result(results_csv_path, row)

        # --- Plotting daily results ---
        df_all_markets = transform_all_markets_to_master_df(df_FCR, df_DAA, df_IDA, df_IDC)
        plotter = MultiMarketPlotter(cfg, df_all_markets, capacity_mwh=capacity, nominal_power_mw=max_power)
        if cfg.plotting.enable_plot_day:
            plotter.plot_day(day, multi_market_optimizer.result_fcr, multi_market_optimizer.result_daa, multi_market_optimizer.result_ida, multi_market_optimizer.result_idc)
        if cfg.plotting.enable_plot_horizon:
            plotter.plot_horizon(day, multi_market_optimizer.result_fcr, multi_market_optimizer.result_daa, multi_market_optimizer.result_ida, multi_market_optimizer.result_idc)
 
    

    # --------------------------------------------------------------
    # Final plot and result
    # --------------------------------------------------------------

    if cfg.plotting.enable_plot_profit_index and plotter is not None:
        plotter.plot_profit_index(csv_path=results_csv_path)

    log.info("Battery Optimizer finished successfully")
    log.info("Final Profit: %.2f", total_profit_for_observed_period)



if __name__ == "__main__":
    main()
