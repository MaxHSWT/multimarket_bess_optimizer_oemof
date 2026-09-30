from __future__ import annotations
import pandas as pd


def transform_all_markets_to_master_df(df_FCR: pd.DataFrame, df_DAA: pd.DataFrame, df_IDA: pd.DataFrame, df_IDC: pd.DataFrame) -> pd.DataFrame:
    """
    Build one combined market dataframe on a 15-minute master index.
    Inputs: df_FCR, df_DAA, df_IDA, df_IDC (preprocessed by load_all_markets_window(...), Expected to have DatetimeIndex and column "price")
    Output: df_all_markets_data (DataFrame with 15-minute master index and columns: price_fcr, price_daa, price_ida, price_idc)
    Rules:
        - IDA/IDC are already expected on 15-min resolution
        - DAA (1h) and FCR (4h) are forward-filled to 15-min resolution
        - Index range is taken from the 15-min master index
    """

    # --- 1) Basic input validation ---
    inputs = {
        "FCR": df_FCR,
        "DAA": df_DAA,
        "IDA": df_IDA,
        "IDC": df_IDC,
    }

    for name, df in inputs.items():
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"df_{name} must be a pandas DataFrame.")
        if df.empty:
            raise ValueError(f"df_{name} is empty.")
        if "price" not in df.columns:
            raise KeyError(f"df_{name} must contain a column 'price'.")
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError(f"df_{name}.index must be a pandas DatetimeIndex.")
        if df.index.has_duplicates:
            raise ValueError(f"df_{name}.index contains duplicate timestamps.")

    # --- 2) Define 15-min master index ---
    # Prefer IDC as master because it is the latest market stage and already 15-min based.
    master_index = df_IDC.index.copy()

    if len(master_index) == 0:
        raise ValueError("IDC master index is empty.")

    # Optional safety check: IDA should match the same 15-min horizon
    if not df_IDA.index.equals(master_index):
        raise ValueError(
            "df_IDA.index does not match df_IDC.index. "
            "Expected both to already be aligned on the same 15-minute horizon."
        )

    # --- 3) Rename price columns clearly ---
    fcr_price = df_FCR[["price"]].rename(columns={"price": "price_fcr"})
    daa_price = df_DAA[["price"]].rename(columns={"price": "price_daa"})
    ida_price = df_IDA[["price"]].rename(columns={"price": "price_ida"})
    idc_price = df_IDC[["price"]].rename(columns={"price": "price_idc"})

    # --- 4) Reindex onto 15-min master ---
    # FCR: 4h -> 15min via forward-fill
    fcr_price_15 = fcr_price.reindex(master_index, method="ffill")

    # DAA: reindex onto 15min master (forward-fill handles coarser DAA blocks, e.g. 1h -> 15min; no-op for native 15min DAA)
    daa_price_15 = daa_price.reindex(master_index, method="ffill")

    # IDA / IDC: already 15min, but align defensively
    ida_price_15 = ida_price.reindex(master_index)
    idc_price_15 = idc_price.reindex(master_index)

    # --- 5) Final combined dataframe ---
    df_all_markets_data = pd.concat(
        [fcr_price_15, daa_price_15, ida_price_15, idc_price_15],
        axis=1,
    )

    # --- 6) Defensive NaN handling ---
    # After proper preprocessing there should normally be no NaNs left.
    # Still, use ffill/bfill as a defensive fallback.
    df_all_markets_data = df_all_markets_data.ffill().bfill()

    # --- 7) Final sanity check ---
    if df_all_markets_data.isna().any().any():
        nan_cols = df_all_markets_data.columns[df_all_markets_data.isna().any()].tolist()
        raise ValueError(
            f"Combined dataframe still contains NaN values in columns: {nan_cols}"
        )

    return df_all_markets_data