# Installation

The model runs on Python 3.11 in a conda environment named `oemof-solph-env`. The
environment is defined in [configs/environment.yml](configs/environment.yml).

## 1. Create the environment

Requires [Miniconda or Anaconda](https://docs.conda.io/). Run all commands from the
project root.

```bash
conda env create -f configs/environment.yml
conda activate oemof-solph-env
```

The environment installs:

| Package | Version | Role |
|---------|---------|------|
| `python` | 3.11 | Runtime |
| `oemof.solph` | 0.6.1 | Energy system modelling framework (builds the MILP) |
| `pyomo` | installed with oemof.solph | Optimization modelling layer |
| `gurobi` | 12.0 | MILP solver (Python package, licence required, see below) |
| `hydra-core` | 1.3.2 | Configuration management (`configs/config.yaml` and command-line overrides) |
| `omegaconf` | 2.3.0 | Configuration backend used by Hydra |
| `pandas` | >= 2.0 | Time series handling |
| `numpy` | >= 2.0 | Numerics |
| `plotly` | >= 6.0 | Interactive HTML plots |

## 2. Solver

The optimization needs a MILP solver. The solver is selected in `configs/config.yaml`
under `optimization.solver`.

This project uses Gurobi with a free academic licence, so the environment installs the
Gurobi package. Academic licences are available at <https://www.gurobi.com/academia/>.

If you don't have a suitable licence, you can use an alternative MILP solver such as CBC.
In that case, install the solver yourself, remove `gurobi` from `configs/environment.yml`
if you don't need it, and set the solver name under `optimization.solver` in
`configs/config.yaml`. Alternative solvers have not been tested with this model.

## 3. Run the model

```bash
python main.py
```

The configuration is read from `configs/config.yaml`. Any parameter can be overridden on
the command line (Hydra syntax), e.g.:

```bash
python main.py simulation.start_date=2025-09-29 simulation.end_date=2025-09-30
python main.py battery.capacity=4.0 plotting.enable_plot_day=False
```

Results and plots are written to the folder set in `plotting.output_dir` (default
`results/`). Hydra additionally writes a log file and a copy of the configuration of each
run to `outputs/`.

## 4. Standalone comparison plots

The scripts `src/visualization/create_*.py` are not part of a run. They compare the daily
profits of several runs from a CSV file. Set `ROOT_PATH` and `CSV_FILENAME` at the top of a
script and run it from the project root, e.g.:

```bash
python -m src.visualization.create_boxplots
```

## Notes

- The model was developed and tested on Windows 11 with the environment above.
- To remove the environment: `conda env remove -n oemof-solph-env`.
