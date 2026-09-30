# src/

Python source code of the multi-market battery optimizer. The entry point `main.py`
(project root) runs the simulation day by day and uses the modules below. All modules are
imported as a package (`src.model`, `src.utils`, `src.visualization`), so run everything
from the project root.

## model/: optimization core

| File | Role |
|------|------|
| `market_optimizer.py` | `MarketOptimizer`: builds and solves the per-market oemof.solph / Pyomo MILP. Contains the FCR optimization (myopic and perfect foresight) and the energy-market optimization (DAA, IDA, IDC), including battery, FCR and FCA constraints. |
| `multi_market_optimizer.py` | `MultiMarketOptimizer`: daily orchestration. Runs the markets in sequence (FCR, DAA, IDA, IDC), passes FCR constraints and fixed positions to the downstream markets, and returns the committed profit and end-of-day SoC. |

## utils/: helper functions

| File | Role |
|------|------|
| `import_and_load_data.py` | Reads and caches the input CSVs, slices the rolling horizon and checks data availability. |
| `get_fcr_constraints.py` | Derives SoC and power limits from the marketed FCR power (PQ rules). |
| `get_fca_constraints.py` | Derives grid power limits from the FCA profile and merges them with the FCR constraints. |
| `calculate_market_dispatch.py` | Builds the net dispatch [MW] and the resulting SoC trajectory [MWh], including efficiencies. |
| `transform_pd_series.py` | Reindexing and resampling of series (e.g. expanding 4 h / 1 h blocks to 15 min). |
| `transform_pd_dataframes.py` | Combines the four market DataFrames onto one 15 min index. |
| `save_optimization_results.py` | Creates `results.csv` and appends one row per simulated day. |
| `plot_helpers.py` | Shared styling and series helpers for the plotter. |

## visualization/: plots

| File | Role |
|------|------|
| `plotter.py` | `MultiMarketPlotter`: day, horizon and profit-index plots, called by `main.py` (see `plotting` in `configs/config.yaml`). |
| `create_boxplots.py` | Standalone script: boxplot of daily profits per model. |
| `create_profit_development_abs.py` | Standalone script: daily and cumulative profit per model. |
| `create_profit_development_avg.py` | Standalone script: 28-day rolling average and cumulative profit per model. |
| `create_profit_development_month.py` | Standalone script: same as `_abs`, with daily axis ticks for short periods (e.g. one month). |

The standalone scripts are not part of a simulation run. They compare several runs: they
read a CSV with a date column (`Index`) and one column of daily profits per model, and
write an `.html` plot next to it. Set `ROOT_PATH` and `CSV_FILENAME` at the top of the
script, then run it, e.g. `python -m src.visualization.create_boxplots`.
