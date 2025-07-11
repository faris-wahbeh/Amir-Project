import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

INITIAL_INVESTMENT = 100.0

@st.cache_data
def load_ranked_returns(file_path: str):
    df = pd.read_csv(file_path)
    df.columns = (
        df.columns
          .str.strip()
          .str.lower()
          .str.replace(r"\s+", "_", regex=True)
    )
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")

    return_cols = sorted(
        [c for c in df.columns if c.startswith("return_rank_")],
        key=lambda c: int(c.split("_")[-1])
    )
    # turn “0.32%” → 0.32
    df[return_cols] = df[return_cols].replace("%", "", regex=True).astype(float)
    return df, return_cols

def calculate_declining_weights(num_positions: int, cash_pct: float) -> np.ndarray:
    total = 1.0 - cash_pct
    ranks = np.arange(num_positions, 0, -1)   # [N, N-1, …, 1]
    return ranks / ranks.sum() * total

def calculate_monthly_net_returns(
    df: pd.DataFrame,
    return_cols: list[str],
    num_positions: int,
    cash_pct: float
) -> pd.Series:
    """
    Returns a Series of net returns (as decimals), one per date:
      net_t = sum_i [ weight_i * (raw_return_i / 100) ]
    """
    w = calculate_declining_weights(num_positions, cash_pct)
    sel = return_cols[:num_positions]
    # df[sel] is in percent-points (e.g. 0.32 → divide by 100 → 0.0032)
    weighted = df[sel].div(100).multiply(w, axis=1)
    return weighted.sum(axis=1)

def get_actual_benchmark_returns() -> list[float]:
    return [
        5.34, 0.16, 1.4, 2.8, 4.98, 5.38, 1.27, 7.16, 0.81, -8.68,
        3.52, -8.29, 8.75, 9.03, 3.26, 5.04, -2.46, 6.89, 3.32, -0.27,
        -6.38, 0.6, 5.64, 0.68, 5.59, -6.47, -14.21, 14.33, 9.07, 3.96,
        6.06, 6.67, -3.01, -3.56, 11.92, 7.29, -2.44, 8.33, -6.72, 5.05,
        -6.17, 6.33, 3.65, 5.27, -2.18, 3.22, -1.75, 0.02, -12.82, -0.39,
        -2.01, -9.02, -9.27, -8.22, 9.11, -2.98, -7.97, 1.83, 4.05, -2.56
    ]

def main():
    st.set_page_config(page_title="Compounded Portfolio", layout="wide")
    st.title("Ranked Portfolio vs Benchmark")

    # 1) Load & clean
    df, return_cols = load_ranked_returns("ranked_returns_top15.csv")

    # 2) Controls
    st.sidebar.header("Settings")
    num_positions = st.sidebar.slider("Number of Positions", 1, 15, 5)
    cash_pct      = st.sidebar.slider("Cash %", 0.0, 100.0, 15.0) / 100.0

    # 3) Compute net returns
    monthly_net = calculate_monthly_net_returns(df, return_cols, num_positions, cash_pct)

    # 4) Show them
    st.subheader("Monthly Net Returns")
    st.dataframe(
        monthly_net
          .mul(100)
          .round(2)
          .astype(str)
          .add("%")
          .to_frame("Net Return")
    )

    # 5) Debug: breakdown per rank
    with st.expander("🔍 Debug weighted contributions"):
        weights = calculate_declining_weights(num_positions, cash_pct)
        sel = return_cols[:num_positions]

        debug = df[sel].copy()
        # raw decimal return for each rank
        for i, col in enumerate(sel):
            debug[f"{col}_dec"]      = debug[col] / 100
            debug[f"{col}_weight"]   = weights[i]
            debug[f"{col}_product"]  = debug[f"{col}_dec"] * weights[i]

        debug["net_return_dec"] = debug[[f"{c}_product" for c in sel]].sum(axis=1)
        debug["net_return_%"]   = (debug["net_return_dec"] * 100).round(3).astype(str) + "%"

        st.dataframe(debug[
            [*sel, 
             *[f"{c}_dec" for c in sel],
             *[f"{c}_weight" for c in sel],
             *[f"{c}_product" for c in sel],
             "net_return_%"
            ]
        ].head(10))

    # 6) Compound
    model = (1 + monthly_net).cumprod() * INITIAL_INVESTMENT
    model.name = "Model Portfolio"

    # 7) Benchmark
    bench = pd.Series(
        get_actual_benchmark_returns()[: len(df)],
        index=df.index
    ).div(100)
    bm = (1 + bench).cumprod() * INITIAL_INVESTMENT
    bm.name = "Benchmark"

    # 8) Plot
    combined = pd.concat([model, bm], axis=1)
    st.subheader("Portfolio Value Over Time")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(combined.index, combined["Model Portfolio"], label="Model", linewidth=2)
    ax.plot(combined.index, combined["Benchmark"],       label="Benchmark", color="orange", linewidth=2)
    ax.set_xlabel("Date"); ax.set_ylabel("Value ($)")
    ax.legend(); ax.grid(True, linestyle="--", alpha=0.5)
    st.pyplot(fig)

    # 9) Metrics
    st.subheader("Model Performance")
    fv = model.iloc[-1]
    tr = (fv - INITIAL_INVESTMENT) / INITIAL_INVESTMENT
    ar = (fv / INITIAL_INVESTMENT) ** (12 / len(model)) - 1
    st.metric("Final Value",       f"${fv:.2f}")
    st.metric("Total Return",      f"{tr:.2%}")
    st.metric("Annualized Return", f"{ar:.2%}")

if __name__ == "__main__":
    main()
