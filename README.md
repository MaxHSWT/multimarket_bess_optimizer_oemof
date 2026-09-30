# Multi-Market Battery Optimizer

This code is part of a Master's thesis at the University of Applied Sciences
Weihenstephan-Triesdorf (HSWT), Germany, and was developed within the research project SmartBattery.
Developer: Max Werner.

Thesis: "Multimarket-Optimierung eines netzgekoppelten Großbatteriespeichers; Entwicklung
eines mehrstufigen Optimierungsmodells in Python zur Vermarktung von netzgekoppelten
Speichern an Spot- und Regelenergiemärkten unter Berücksichtigung von
Netzzugangsrestriktionen" (2026).

The code is only complete together with the thesis, which contains the model description,
the methodology and the discussion of results.

This repository demonstrates how a sequential multi-market optimization of a grid-connected
battery energy storage system (BESS) can be implemented with
[oemof.solph](https://oemof-solph.readthedocs.io/). The battery is marketed on several
German electricity markets in their chronological order on a daily basis:

1. FCR: Frequency Containment Reserve (capacity market, 4 h blocks)
2. DAA: Day-Ahead Auction (energy, 1 h or 15 min resolution)
3. IDA: Intraday Auction (energy, 15 min resolution)
4. IDC: Intraday Continuous (energy, 15 min resolution, price index such as ID1)

Each market is optimized in turn. The positions of the previous markets are handed over
as fixed boundaries to the next market. Each stage is formulated as a MILP with
oemof.solph / Pyomo. Grid restrictions can be modelled through optional FCA (Flexible
Connection Agreement) power limits.

The model runs as a rolling horizon over a configurable simulation period: every day is
optimized over a horizon of `lookahead_hours`, but only the first 24 h are committed. The
end-of-day state of charge is carried over to the next day. Two modes are available per
market: myopic (optimization on the current market only) and perfect foresight
(co-optimization with the downstream markets).

## Demo data

The repository contains only freely invented, deliberately simple demo data, so the model
can be run directly and its results can be followed by eye. The real market data used in
the thesis is not included. See [data/read_me.md](data/read_me.md) for the data structure
and for how to use your own data.

## Repository layout

| Path | Purpose |
|------|---------|
| `main.py` | Entry point, runs the rolling-horizon simulation |
| `configs/` | Configuration (`config.yaml`) and conda environment (`environment.yml`) |
| `src/model/` | Optimization core (oemof.solph / Pyomo models and daily orchestration) |
| `src/utils/` | Data import, transformations, constraints, result export |
| `src/visualization/` | Plots of a run and standalone scripts for comparing runs |
| `data/` | Demo input data (market prices and FCA grid limits) |
| `results/` | Results of a run (`results.csv` and interactive `.html` plots) |
| `outputs/` | Hydra log files and config copies of each run |
| `requirements.md` | Installation and solver setup |

Each folder contains a `read_me.md` with details.

## Quick start

A MILP solver is required (Gurobi by default if available, alternatively an open-source solver like CBC), see
[requirements.md](requirements.md).

```bash
conda env create -f configs/environment.yml
conda activate oemof-solph-env
python main.py
```

As delivered, `configs/config.yaml` runs the demo data from 2025-09-29 to 2025-10-02. The
results are written to `results/`. Any parameter can be overridden on the command line
(Hydra syntax), e.g.:

```bash
python main.py battery.capacity=4.0 optimization.rolling_horizon.lookahead_hours=24
```

## Configuration

All model parameters (battery, simulation period, input data, optimization mode, grid
constraints, plotting) are set in [configs/config.yaml](configs/config.yaml). See
[configs/read_me.md](configs/read_me.md) for a description of every section.

## License

This code is released under the MIT License, see [LICENSE](LICENSE). If you use it in your
own work, please cite the thesis mentioned above.
