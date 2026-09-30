# data/

Input data for the model.

## Demo data only!

The files in this folder are demo datasets, not real market data. They are freely invented
and deliberately primitive (flat price levels, a few isolated spikes, simple step patterns).
They exist so that you can

- run the model end-to-end right after cloning the repository,
- check that your setup works, and
- understand the optimization results by eye: for every result you should be able to
  follow why the battery charged or discharged at a given time.

They do not represent real prices, real grid limits or any real market behaviour.
Revenues computed from them are meaningless in absolute terms.
To run proper simulations, supply your own data in the formats described at the bottom.

## Files

| `market-data/demo_da-id-prices.csv` | Energy-market prices DAA / IDA / IDC | 15 min | 2025-09-29 00:00 to 2025-10-03 23:45 (5 days) |
| `market-data/demo_fcr-prices.csv` | FCR capacity prices | 4 h blocks | 2025-09-29 to 2025-10-03 (5 days x 6 blocks) |
| `fca-data/demo_fca-boundaries.csv` | Grid power limits (FCA) | 15 min day profile per month | one generic day per month |

The files are grouped in two subfolders that are referenced in `config.yaml`:

- `market-data/` holds the price data (`data.subfolder`).
- `fca-data/` holds the grid power limits (`grid_constraints.subfolder`).

The filenames and column names in `config.yaml` must match the files you want to use.

### Market abbreviations

| FCR | Frequency Containment Reserve (Primaerregelleistung), 4 h capacity blocks |
| DAA | Day-Ahead Auction, hourly prices (given on the 15 min grid, see below) |
| IDA | Intraday Auction 1, 15 min |
| IDC | Intraday Continuous Index (e.g. ID1, ID3), 15 min |

## File formats

General conventions for all CSVs:

- Separator `;`, decimal mark `,` (German export format).
  The model reads every file with `decimal=","`, so use `,` for non-integer values.
- Prices are in EUR/MWh (energy markets) or EUR/MW (FCR capacity).
- More data than the simulated period may be needed. The data must cover the full window
  from `simulation.start_date` to `simulation.end_date` plus the look-ahead
  (`optimization.rolling_horizon.lookahead_hours`) and, if the timezone of the data differs
  from `simulation.timezone`, an additional margin for the timezone shift. For the demo files
  this means: `end_date` at most 2025-10-03 with `lookahead_hours: 24`, and at most
  2025-10-02 with `lookahead_hours: 48`. If a run fails because of missing data, first check
  whether the files cover the period including the look-ahead.

### market-data/demo_da-id-prices.csv: energy markets

| Column | Meaning |
|--------|---------|
| `timestamp` | Timezone-aware start of the 15 min interval, e.g. `2025-09-29 00:00:00+02:00`. UTC, CET and CEST offsets are accepted and are converted to the timezone set in `config.yaml` (`simulation.timezone`). |
| `price_daa` | Day-ahead price [EUR/MWh]. Hourly product: only filled at full hours, empty in between (the model expands it to the 15 min grid). |
| `price_ida` | Intraday-auction price [EUR/MWh], one value per 15 min. |
| `price_idc` | Intraday-continuous price [EUR/MWh], one value per 15 min. |

Column names are free. The ones actually used are selected in `config.yaml` under
`data.markets.<MARKET>.price_column`.

### market-data/demo_fcr-prices.csv: FCR capacity market

The layout follows the export of [regelleistung.net](https://www.regelleistung.net)
(section "Daten", FCR results). The model only reads the three columns marked "yes" below;
the others are kept for format compatibility and can be left as they are.

| Column | Used | Meaning |
|--------|:----:|---------|
| `DATE_FROM` | yes | Delivery date, format `DD.MM.YYYY`. |
| `PRODUCTNAME` | yes | 4 h block: `NEGPOS_00_04`, `NEGPOS_04_08`, `NEGPOS_08_12`, `NEGPOS_12_16`, `NEGPOS_16_20`, `NEGPOS_20_24`. Together with `DATE_FROM` it forms the timestamp of the block. |
| `GERMANY_SETTLEMENTCAPACITY_PRICE_[EUR/MW]` | yes | Capacity price of the block [EUR/MW]. |
| other columns | no | Not evaluated by the model. |

Each day needs exactly 6 rows (one per block).

### fca-data/demo_fca-boundaries.csv: grid power limits (FCA)

Only needed if `grid_constraints.apply_fca: True`. Flexible Connection Agreements limit the
charge/discharge power at the grid connection point depending on time of day and month.

- 96 rows = one day in 15 min steps, `Index` 0 to 95 (0 = 00:00, 95 = 23:45).
- For each month two columns: `cha_<month>` (charging limit) and `discha_<month>`
  (discharging limit). Month suffixes are the German abbreviations:
  `jan, feb, mae, apr, mai, jun, jul, aug, sep, okt, nov, dez`.
- Values are fractions [0..1] of the technical power (`battery.max_charge_power` /
  `battery.max_discharge_power`): `1` = full power available, `0` = no power allowed.
  The same day profile is applied to every day of the respective month.

Note: the demo file is `;`-separated, so set `grid_constraints.separator: ";"`
(the default in `config.yaml` may be a tab).

## Using your own data

To run the model on real data, place your price files in `data/market-data/` and your FCA file in
`data/fca-data/` (or other subfolders referenced via `data.subfolder` and
`grid_constraints.subfolder`), keep the formats described above, and point the `data` and
`grid_constraints` sections of `config.yaml` to them. Real market data (e.g. from
EPEX SPOT, regelleistung.net or your grid operator) is usually subject to licence terms.
