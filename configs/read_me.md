# configs/

Configuration for the model and its runtime environment.

## Files

| File | Format | Purpose |
|------|--------|---------|
| `config.yaml` | YAML (Hydra) | Central configuration for every model run |
| `environment.yml` | YAML (conda) | Conda environment definition (`oemof-solph-env`) |

## config.yaml

Loaded by Hydra in `main.py`. Every value can be overridden from the command line, e.g.
`python main.py battery.capacity=4.0`. As delivered, the file is set up to run with the
demo datasets in [../data/](../data/read_me.md). Sections:

- `battery`: physical battery parameters: capacity [MWh], charge/discharge power [MW],
  charge/discharge efficiency, initial SoC, max daily cycles, and the marketable FCR
  power fraction.
- `simulation`: `start_date`, `end_date` (inclusive) and `timezone`. Input data must
  cover this window plus the look-ahead.
- `data`: location and parsing of the energy- and capacity-market price CSVs:
  `base_dir`, `subfolder`, and per market (`FCR`, `DAA`, `IDA`, `IDC`) filename,
  separator, frequency and column names.
- `optimization`: `solver`, rolling-horizon settings (`commit_hours`, `lookahead_hours`)
  and per-market flags (`run_*` to enable a market, `perfect_foresight` to co-optimize
  with downstream markets instead of optimizing myopically).
- `grid_constraints`: FCA settings: `apply_fca` toggle and the path/separator of the
  FCA boundary CSV.
- `plotting` / `plot`: which plots to generate, the output directory (`output_dir`),
  image export settings, and general plot styling (colors, opacity, line widths).

## environment.yml

Conda environment definition used to reproduce the runtime. See
[../requirements.md](../requirements.md) for setup instructions.
