from pathlib import Path
import pandas as pd
import plotly.graph_objects as go

"""Independent script to plot boxplots from given results
    - Expects csv-file with results
    - Plots one boxplot per model-results column
    - Returns html-file with interactive boxplots
    - Can be triggered independently and directly here"""

# ============================================================
# SETTINGS
# ============================================================

ROOT_PATH = "results"            # folder of the input CSV, relative to the project root
CSV_FILENAME = "example.csv"

DATE_COLUMN = "Index"          # first column containing the date, e.g. Index
Y_AXIS_TITLE = "Profit [€/MW/d]"
OUTPUT_FILENAME = "boxplot_daily-profits-per-model.html"

CSV_SEPARATOR = ";"            # ";" for German-format CSV files
DECIMAL_SEPARATOR = ","        # "," for comma decimal numbers


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
        # If the first column has a different name, use it as the date column
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

    return df


def prepare_long_format(df: pd.DataFrame) -> pd.DataFrame:
    """Convert the wide model columns into a long format for Plotly."""

    model_columns = [col for col in df.columns if col != DATE_COLUMN]

    df_long = df.melt(
        id_vars=DATE_COLUMN,
        value_vars=model_columns,
        var_name="Modell",
        value_name="Profit",
    )

    df_long = df_long.dropna(subset=["Profit"])

    return df_long


def _resolve_label_overlaps(y_positions: list[float], min_gap: float) -> list[float]:
    """Iteratively push label y-positions apart until no overlaps remain.
    Labels must be passed sorted ascending; adjusted positions are returned sorted ascending."""
    pos = list(y_positions)
    for _ in range(500):
        changed = False
        for j in range(len(pos) - 1):
            gap = pos[j + 1] - pos[j]
            if gap < min_gap:
                shift = (min_gap - gap) / 2
                pos[j] -= shift
                pos[j + 1] += shift
                changed = True
        if not changed:
            break
    return pos


def create_boxplot(df_long: pd.DataFrame, output_path: Path) -> None:
    """Create and save the interactive Plotly boxplot."""

    models = df_long["Modell"].unique().tolist()
    # Muted sequential palette: blue → violet → red → orange → yellow → green
    colors = [
        "#4878A8",  # muted blue
        "#7B5EA7",  # muted violet
        "#B55353",  # muted red
        "#C97B3A",  # muted orange
        "#C4AC3A",  # muted yellow
        "#4E8C5A",  # muted green
    ]

    # Estimate minimum label gap in data units (font 10px, usable plot height ~600px)
    all_vals = df_long["Profit"].dropna()
    y_range = all_vals.max() - all_vals.min()
    min_gap = (y_range / 600) * 14  # 14px per label line

    fig = go.Figure()

    annotations = []

    for i, model in enumerate(models):
        data = df_long[df_long["Modell"] == model]
        color = colors[i % len(colors)]

        fig.add_trace(
            go.Box(
                y=data["Profit"],
                x=data["Modell"],
                name=model,
                marker_color=color,
                width=0.4,                 # box width (0-1, default ~0.25)
                boxmean=True,              # additionally shows the mean
                boxpoints="outliers",      # shows outlier points
                hovertemplate=(
                   "Profit: %{y:.2f} €/MW"
                   "<extra></extra>"
                )
            )
        )

        # Compute statistics and place them as annotations to the right of the box
        vals = data["Profit"].dropna()
        n = len(vals)
        stats = [
            ("Min",    vals.min()),
            ("Q1",     vals.quantile(0.25)),
            ("Median", vals.median()),
            ("Mean",   vals.mean()),
            ("Q3",     vals.quantile(0.75)),
            ("Max",    vals.max()),
        ]  # sorted ascending by value for the overlap resolver

        # n label above the box
        annotations.append(dict(
            x=i,
            y=vals.max(),
            xref="x",
            yref="y",
            text=f"n={n}",
            showarrow=False,
            xanchor="center",
            yanchor="bottom",
            yshift=6,
            font=dict(color=color, size=10),
        ))

        y_original = [v for _, v in stats]
        y_adjusted = _resolve_label_overlaps(y_original, min_gap)

        # x position: categorical axis → index i, slightly right of the box (width 0.5 → edge at 0.25)
        x_pos = i + 0.28

        for (label, value), y_pos in zip(stats, y_adjusted):
            annotations.append(dict(
                x=x_pos,
                y=y_pos,
                xref="x",
                yref="y",
                text=f"{label}: {value:.1f}",
                showarrow=False,
                xanchor="left",
                yanchor="middle",
                font=dict(color=color, size=10),
            ))

    # Colored bold tick labels for x-axis (HTML supported via ticktext)
    tick_labels = [
        f"<b><span style='color:{colors[i % len(colors)]}'>{model}</span></b>"
        for i, model in enumerate(models)
    ]

    fig.update_layout(
        title="Verteilung der täglichen Profite je Modus und Modell",
        xaxis_title="Variante",
        yaxis_title=Y_AXIS_TITLE,
        template="plotly_white",
        hovermode="closest",
        boxmode="group",
        showlegend=False,
        hoverlabel=dict(
            bgcolor="rgba(255, 255, 255, 0.5)",
            font_size=13,
        ),
        annotations=annotations,
        height=750,
        margin=dict(l=70, r=40, t=80, b=70),
    )

    fig.update_yaxes(
        zeroline=True,
        zerolinewidth=1,
        automargin=True,
        showgrid=True,
        gridcolor="lightgray",
        gridwidth=1,
        showline=True,
        linecolor="darkgray",
        linewidth=1,
    )

    fig.update_xaxes(
        showgrid=True,
        gridcolor="lightgray",
        gridwidth=1,
        showline=True,
        linecolor="darkgray",
        linewidth=1,
        tickvals=models,
        ticktext=tick_labels,
    )

    export_config = {
        "toImageButtonOptions": {
            "format": "jpeg",
            "filename": output_path.stem,
            "width": 1400,
            "height": 750,
            "scale": 3,
        }
    }

    fig.write_html(str(output_path), include_plotlyjs="cdn", config=export_config)
    print(f"Boxplot saved to: {output_path}")


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    csv_path = project_root / ROOT_PATH / CSV_FILENAME
    output_path = csv_path.parent / OUTPUT_FILENAME

    df = load_results_csv(csv_path)
    df_long = prepare_long_format(df)
    create_boxplot(df_long, output_path)


if __name__ == "__main__":
    main()