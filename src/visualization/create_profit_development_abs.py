from pathlib import Path
import pandas as pd
import plotly.graph_objects as go

"""Independent script to plot profit development from given results.
    - Expects CSV-file with daily profits per model (same format as create_boxplots.py)
    - Plots daily profit (solid lines, left y-axis) per model
    - Plots cumulative profit (dashed lines, right y-axis) per model
    - Same color per model for both lines
    - Returns HTML-file saved next to the input CSV
    - Can be triggered independently and directly here"""

# ============================================================
# SETTINGS
# ============================================================

ROOT_PATH = "results"            # folder of the input CSV, relative to the project root
CSV_FILENAME = "example.csv"

DATE_COLUMN = "Index"                             # first column containing the date
Y_AXIS_LEFT_TITLE = "Täglicher Profit [€/MW/d]"  # left y-axis
OUTPUT_FILENAME = "profit_development-per-model_abs.html"

CSV_SEPARATOR = ";"            # ";" for German-format CSV files
DECIMAL_SEPARATOR = ","        # "," for comma decimal numbers

# Muted sequential palette: blue → violet → red → orange → yellow → green
COLORS = [
    "#4878A8",  # muted blue
    "#7B5EA7",  # muted violet
    "#B55353",  # muted red
    "#C97B3A",  # muted orange
    "#C4AC3A",  # muted yellow
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


def create_profit_development(df: pd.DataFrame, output_path: Path) -> None:
    """Create and save the interactive profit development plot.

    Left y-axis: daily profit (solid line per model)
    Right y-axis: cumulative profit (dashed line per model)
    """

    model_columns = [col for col in df.columns if col != DATE_COLUMN]
    dates = df[DATE_COLUMN]

    # x-axis tick configuration: one tick per month with German month names
    GERMAN_MONTHS = [
        "Januar", "Februar", "März", "April", "Mai", "Juni",
        "Juli", "August", "September", "Oktober", "November", "Dezember",
    ]

    # Collect first day of every month in the date range
    month_starts = pd.date_range(
        start=dates.min().replace(day=1),
        end=dates.max(),
        freq="MS",
    )
    tick_vals = [d.strftime("%Y-%m-%d") for d in month_starts]
    tick_text = [GERMAN_MONTHS[d.month - 1] for d in month_starts]

    fig = go.Figure()

    for i, model in enumerate(model_columns):
        color = COLORS[i % len(COLORS)]
        daily = df[model]

        # Daily profit — solid line, left y-axis
        fig.add_trace(go.Scatter(
            x=dates,
            y=daily,
            name=model,
            mode="lines",
            line=dict(color=color, width=2),
            yaxis="y",
            hovertemplate=(
                "<b>%{x|%d.%m.%Y}</b><br>"
                "Tagesprofit: %{y:.2f}<br>"
                "<extra>" + model + "</extra>"
            ),
        ))

    fig.update_layout(
        title="Profitentwicklung je Modus und Modell",
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
            linewidth=1.5,
            mirror=False,
            ticks="outside",
            tickcolor="darkgray",
            tickvals=tick_vals,
            ticktext=tick_text,
        ),
        yaxis=dict(
            title=Y_AXIS_LEFT_TITLE,
            showgrid=True,
            gridcolor="lightgray",
            gridwidth=0.5,
            zeroline=True,
            zerolinecolor="lightgray",
            zerolinewidth=1.0,
            showline=True,
            linecolor="darkgray",
            linewidth=1.5,
            automargin=True,
        ),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.1,
            xanchor="left",
            x=0,
            entrywidth=1 / 6,        # 1/6 of figure width → 6 items per row, spans full width
            entrywidthmode="fraction",
            font=dict(size=12),
        ),
        width=1400,
        height=800,
        margin=dict(l=70, r=70, t=80, b=130),
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
