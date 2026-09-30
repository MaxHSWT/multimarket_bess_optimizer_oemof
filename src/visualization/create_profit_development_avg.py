import math
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go

"""Independent script to plot profit development from given results — 28-day centred rolling average.
    - Expects CSV-file with daily profits per model (same format as create_boxplots.py)
    - Plots 28-day centred rolling average of daily profit (solid lines, left y-axis) per model
      → an optional, already aggregated column (NO_SMOOTHING_COLUMN) is plotted without smoothing
    - For days at the edges of the timeline the rolling window uses all available surrounding
      days within the timeline (centred, min_periods=1)
    - Plots cumulative profit based on original daily values (dashed lines, right y-axis) per model
    - Plots monthly average as dotted step-line (left y-axis) per model
    - Same color per model for all lines
    - Returns HTML-file saved next to the input CSV
    - Can be triggered independently and directly here"""

# ============================================================
# SETTINGS
# ============================================================

ROOT_PATH = "results"            # folder of the input CSV, relative to the project root
CSV_FILENAME = "example.csv"

DATE_COLUMN = "Index"                              # first column containing the date
Y_AXIS_LEFT_TITLE = "28-Tage-Ø Profit [€/MW/d]"  # left y-axis
Y_AXIS_RIGHT_TITLE = "Kumulierter Profit [€/MW]"  # right y-axis
OUTPUT_FILENAME = "profit_development-per-model-avg28.html"

CSV_SEPARATOR = ";"     # ";" for German-format CSV files
DECIMAL_SEPARATOR = ","  # "," for comma decimal numbers

# Optional: name of a column that is already aggregated and should not be smoothed (None = smooth all)
NO_SMOOTHING_COLUMN = None

ROLLING_WINDOW = 28  # days

# Muted sequential palette: blue → violet → red → orange → yellow → green
COLORS = [
    "#4878A8",  # muted blue
    "#7B5EA7",  # muted violet
    "#B55353",  # muted red
    "#C97B3A",  # muted orange
    "#C4AC3A",  # muted light green
    "#4E8C5A",  # muted green
]


# ============================================================
# FUNCTIONS
# ============================================================

def load_results_csv(csv_path: Path) -> pd.DataFrame:
    """Load the results CSV and validate its basic structure."""

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(
        csv_path,
        sep=CSV_SEPARATOR,
        decimal=DECIMAL_SEPARATOR,
    )

    if df.empty:
        raise ValueError("The CSV file is empty.")

    if DATE_COLUMN not in df.columns:
        first_col = df.columns[0]
        df = df.rename(columns={first_col: DATE_COLUMN})

    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], format="%d.%m.%Y", errors="coerce")

    invalid_count = df[DATE_COLUMN].isna().sum()
    if invalid_count > 0:
        print(f"Warning: {invalid_count} row(s) with an invalid or empty date were ignored.")
        df = df.dropna(subset=[DATE_COLUMN])

    model_columns = [col for col in df.columns if col != DATE_COLUMN]

    if not model_columns:
        raise ValueError("No model columns found.")

    for col in model_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if df[model_columns].isna().all().any():
        invalid_cols = df[model_columns].columns[df[model_columns].isna().all()].tolist()
        raise ValueError(f"These model columns contain no numeric values: {invalid_cols}")

    df = df.sort_values(DATE_COLUMN).reset_index(drop=True)

    return df


def rolling_avg(series: pd.Series) -> pd.Series:
    """Centred 28-day rolling average.

    At the edges of the timeline the window shrinks symmetrically and uses
    all available surrounding days that are still within the timeline
    (min_periods=1 ensures no NaN at the boundaries).
    """
    return series.rolling(window=ROLLING_WINDOW, center=True, min_periods=1).mean()


def create_profit_development(df: pd.DataFrame, output_path: Path) -> None:
    """Create and save the interactive profit development plot (28-day smoothing).

    Left y-axis:  28-day average of the daily profit (solid line per model)
                  + monthly average as dotted step line
    Right y-axis: cumulative profit based on the original daily values
    """

    model_columns = [col for col in df.columns if col != DATE_COLUMN]
    dates = df[DATE_COLUMN]

    # x-axis tick configuration: one tick per month with German month names
    GERMAN_MONTHS = [
        "Januar", "Februar", "März", "April", "Mai", "Juni",
        "Juli", "August", "September", "Oktober", "November", "Dezember",
    ]

    month_starts = pd.date_range(
        start=dates.min().replace(day=1),
        end=dates.max(),
        freq="MS",
    )
    tick_vals = [d.strftime("%Y-%m-%d") for d in month_starts]
    tick_text = [GERMAN_MONTHS[d.month - 1] for d in month_starts]

    # --- Axis alignment: y=0 shared, fixed ratio 250 left ≡ 50 000 right (factor 200) ---
    AXIS_RATIO = 200  # 1 unit left = 200 units right
    _all_left: list[float] = []
    _all_right: list[float] = []
    for _m in model_columns:
        _d = df[_m]
        _s = _d if _m == NO_SMOOTHING_COLUMN else rolling_avg(_d)
        _all_left.extend(_s.dropna().tolist())
        _all_right.extend(_d.cumsum().tolist())

    left_step = 250  # fixed: one gridline every 250 €/MW/d
    RIGHT_STEP = left_step * AXIS_RATIO  # e.g. 250 left → 50 000 right

    _ticks_above = math.ceil(max(max(_all_left) / left_step, max(_all_right) / RIGHT_STEP))
    _ticks_below = math.ceil(max(-min(_all_left) / left_step, -min(_all_right) / RIGHT_STEP))
    left_range = [-_ticks_below * left_step, _ticks_above * left_step]
    right_range = [-_ticks_below * RIGHT_STEP, _ticks_above * RIGHT_STEP]

    fig = go.Figure()

    for i, model in enumerate(model_columns):
        color = COLORS[i % len(COLORS)]
        daily = df[model]
        cumulative = daily.cumsum()

        # Apply 28-day centred rolling average, skip for already-aggregated column
        if model == NO_SMOOTHING_COLUMN:
            daily_smoothed = daily
            smoothing_label = ""
        else:
            daily_smoothed = rolling_avg(daily)
            smoothing_label = " (28-Tage-Ø)"

        # 28-day smoothing of the daily profit — solid line, left y-axis
        fig.add_trace(go.Scatter(
            x=dates,
            y=daily_smoothed,
            name=model,
            mode="lines",
            line=dict(color=color, width=3),
            yaxis="y",
            hovertemplate=(
                "<b>%{x|%d.%m.%Y}</b><br>"
                f"Tagesprofit{smoothing_label}: %{{y:.2f}}<br>"
                "<extra>" + model + "</extra>"
            ),
        ))

        # Cumulative profit (original) — dashed line, right y-axis
        fig.add_trace(go.Scatter(
            x=dates,
            y=cumulative,
            name=f"{model} (kum.)",
            mode="lines",
            line=dict(color=color, width=2, dash="dash"),
            opacity=0.6,
            xaxis="x",
            yaxis="y2",
            hovertemplate=(
                "<b>%{x|%d.%m.%Y}</b><br>"
                "Kum. Profit: %{y:.2f}<br>"
                "<extra>" + model + " kumuliert</extra>"
            ),
        ))

    fig.update_layout(
        title="Profitentwicklung je Modus und Modell (28-Tage-Glättung)",
        template="plotly_white",
        hovermode="x",
        xaxis=dict(
            title="Datum",
            range=[dates.min(), dates.max()],
            showgrid=True,
            gridcolor="lightgray",
            gridwidth=1,
            showline=True,
            linecolor="darkgray",
            linewidth=1,
            mirror=False,
            ticks="outside",
            tickcolor="darkgray",
            tickvals=tick_vals,
            ticktext=tick_text,
        ),
        xaxis2=dict(
            visible=False,
            showline=False,
            matches="x",
        ),
        yaxis=dict(
            title=Y_AXIS_LEFT_TITLE,
            range=left_range,
            dtick=left_step,
            tick0=0,
            tickmode="linear",
            showgrid=True,
            gridcolor="lightgray",
            gridwidth=0.5,
            zeroline=True,
            zerolinecolor="darkgray",
            zerolinewidth=1.5,
            showline=True,
            linecolor="darkgray",
            linewidth=1.5,
            automargin=True,
        ),
        yaxis2=dict(
            title=Y_AXIS_RIGHT_TITLE,
            overlaying="y",
            anchor="x",
            side="right",
            range=right_range,
            dtick=RIGHT_STEP,
            tick0=0,
            tickmode="linear",
            showgrid=True,
            gridcolor="lightgray",
            gridwidth=1,
            griddash="dash",
            zeroline=False,
            showline=True,
            linecolor="darkgray",
            linewidth=1,
            automargin=True,
        ),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.1,
            xanchor="left",
            x=0.0,
            entrywidth=1 / 4,
            entrywidthmode="fraction",
            font=dict(size=12),
        ),
        width=1400,
        height=800,
        margin=dict(l=70, r=100, t=80, b=130),
    )

    export_config = {
        "toImageButtonOptions": {
            "format": "jpeg",
            "filename": output_path.stem,
            "width": 1400,
            "height": 800,
            "scale": 3,
        }
    }

    fig.write_html(str(output_path), include_plotlyjs="cdn", config=export_config)
    print(f"Profit development plot saved to: {output_path}")


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    csv_path = project_root / ROOT_PATH / CSV_FILENAME
    output_path = csv_path.parent / OUTPUT_FILENAME

    df = load_results_csv(csv_path)
    create_profit_development(df, output_path)


if __name__ == "__main__":
    main()
