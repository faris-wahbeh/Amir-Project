import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Constants
INITIAL_INVESTMENT = 100.0

@st.cache_data
def load_ranked_returns(file_path: str):
    """Load CSV, clean column names, parse dates, convert % strings → floats."""
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

    df[return_cols] = (
        df[return_cols]
          .replace("%", "", regex=True)
          .astype(float)
    )
    return df, return_cols

def calculate_declining_weights(num_positions: int, cash_pct: float) -> np.ndarray:
    """Return weights for ranks 1…N that sum to (1 − cash_pct)."""
    total = 1.0 - cash_pct
    ranks = np.arange(num_positions, 0, -1)  # e.g. [4,3,2,1]
    return ranks / ranks.sum() * total

def get_actual_benchmark_returns() -> list[float]:
    """Hard-coded monthly benchmark returns (in percent)."""
    return [
        5.34, 0.16, 1.4, 2.8, 4.98, 5.38, 1.27, 7.16, 0.81, -8.68,
        3.52, -8.29, 8.75, 9.03, 3.26, 5.04, -2.46, 6.89, 3.32, -0.27,
        -6.38, 0.6, 5.64, 0.68, 5.59, -6.47, -14.21, 14.33, 9.07, 3.96,
        6.06, 6.67, -3.01, -3.56, 11.92, 7.29, -2.44, 8.33, -6.72, 5.05,
        -6.17, 6.33, 3.65, 5.27, -2.18, 3.22, -1.75, 0.02, -12.82, -0.39,
        -2.01, -9.02, -9.27, -8.22, 9.11, -2.98, -7.97, 1.83, 4.05, -2.56
    ]

def main():
    st.set_page_config(page_title="Compounding Weighted Portfolio", layout="wide")
    st.title("Ranked Portfolio vs Benchmark")

    # 1) load data
    df, return_cols = load_ranked_returns("ranked_returns_top15.csv")

    # 2) user inputs
    st.sidebar.header("Settings")
    num_positions = st.sidebar.slider("Number of Positions", 1, 15, 4)
    cash_pct      = st.sidebar.slider("Cash %", 0.0, 100.0, 0.0) / 100.0

    # 3) compute weights
    weights = calculate_declining_weights(num_positions, cash_pct)
    selected = return_cols[:num_positions]

    # debug: display weights
    st.write("**Weights for positions 1→N:**", dict(zip(selected, weights)))
    st.write("**Sum of weights (should be 1 − cash %):**", weights.sum())

    # 4) build per-position factors (1 + return), weight & sum → portfolio factor
    factors = (
        df[selected]
        .div(100)         # percent → decimal
        .add(1)           # return → growth factor
        .multiply(weights, axis=1)
        .sum(axis=1)      # sum across the top-N positions
    )

    # 5) extract monthly returns if you need them
    portfolio_returns = factors - 1

    # 6) compound via cumprod
    model_series = factors.cumprod().mul(INITIAL_INVESTMENT)
    model_series.name = "Model Portfolio"

    # 7) benchmark compounding
    bench_percents = get_actual_benchmark_returns()
    bench_returns  = pd.Series(bench_percents[:len(df)], index=df.index) / 100
    bench_factors  = bench_returns.add(1)
    actual_series  = bench_factors.cumprod().mul(INITIAL_INVESTMENT)
    actual_series.name = "Benchmark"

    # 8) plot
    combined = pd.concat([model_series, actual_series], axis=1)
    st.subheader("Portfolio Value Over Time")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(combined.index, combined["Model Portfolio"], label="Model", linewidth=2)
    ax.plot(combined.index, combined["Benchmark"],       label="Benchmark", linewidth=2, color="orange")
    ax.set_xlabel("Date"); ax.set_ylabel("Value ($)")
    ax.set_title("Compounded Portfolio vs Benchmark")
    ax.legend(); ax.grid(True, linestyle="--", alpha=0.5)
    st.pyplot(fig)

    # 9) metrics
    st.subheader("Model Performance")
    final_val = model_series.iloc[-1]
    total_ret = (final_val - INITIAL_INVESTMENT) / INITIAL_INVESTMENT
    ann_ret   = (final_val / INITIAL_INVESTMENT) ** (12 / len(model_series)) - 1

    st.metric("Final Value",       f"${final_val:.2f}")
    st.metric("Total Return",      f"{total_ret:.2%}")
    st.metric("Annualized Return", f"{ann_ret:.2%}")

    # 10) show monthly returns
    with st.expander("Monthly Portfolio Returns"):
        st.dataframe((portfolio_returns * 100).round(2).astype(str) + "%")

if __name__ == "__main__":
    main()
