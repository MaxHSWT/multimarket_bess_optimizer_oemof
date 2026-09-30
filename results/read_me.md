# results/

Optimization outputs: a `results.csv` per run plus interactive `.html` plots.

## Output files of a run

All files are written to `plotting.output_dir` in `config.yaml` (default: `results/`).
Existing files with the same name are overwritten.

| File | When | Content |
|------|------|---------|
| `results.csv` | always | Daily results table, one row per simulated day (see below) |
| `plot_<date>_day.html` | `enable_plot_day: True` | One per day: dispatch per market, prices, SoC and profit over the committed 24 h |
| `plot_<date>_horizon.html` | `enable_plot_horizon: True` | One per day: same plot over the full look-ahead horizon |
| `plot_profit-index.html` | `enable_plot_profit_index: True` | One per run: cumulative profit over the simulation period, built from `results.csv` |

The `.html` plots open in any browser. The camera icon in the plot toolbar exports them as
an image, using the settings in `plotting.export`.

In addition, Hydra writes a log file and a copy of the used configuration for every run to
`outputs/<date>/<time>/` in the project root.

## results.csv

Written by `save_optimization_results.py`, one row per committed day. The first columns
are the headline figures:

| Column | Meaning |
|--------|---------|
| `date` | Committed day |
| `daily_profit` | Total profit of the day across all markets [EUR] |
| `end_of_day_soc` | State of charge carried to the next day [MWh] |
| `total_profit_so_far` | Cumulative profit over the simulation window [EUR] |

The remaining columns break the result down per market (`*_fcr`, `*_daa`, `*_ida`,
`*_idc`): marketed power, bought/sold energy, per-stage profit (`*_horizon` over the full
look-ahead, `*_commit` over the committed 24 h) and SoC trajectories.

The contents of this folder (except this file) are not tracked by git (see `.gitignore`).
