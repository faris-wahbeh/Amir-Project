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
        [col for col in df.columns if col.startswith("return_rank_")],
        key=lambda c: int(c.split("_")[-1])
    )

    # "0.32%" → 0.32
    df[return_cols] = df[return_cols].replace("%", "", regex=True).astype(float)
    return df, return_cols

def calculate_declining_weights(num_positions: int, cash_pct: float) -> np.ndarray:
    """
    Generate weights that decline linearly from rank 1 to rank N,
    summing to (1 - cash_pct).
    """
    total_weight = 1.0 - cash_pct
    ranks = np.arange(num_positions, 0, -1)  # e.g. [5,4,3,2,1]
    weights = ranks / ranks.sum() * total_weight
    return weights

def compute_portfolio_returns(
    df: pd.DataFrame,
    return_cols: list[str],
    num_positions: int,
    cash_pct: float
) -> pd.Series:
    """
    For each month, take the top-N returns, convert % → decimal,
    multiply by their declining weights, and sum.
    Returns a Series of monthly portfolio returns (as decimals).
    """
    weights = calculate_declining_weights(num_positions, cash_pct)
    selected = return_cols[:num_positions]
    # df[selected] holds percentages like 0.32 → divide by 100 to get 0.0032
    weighted = df[selected].div(100).multiply(weights, axis=1)
    return weighted.sum(axis=1)

def get_actual_benchmark_returns() -> list[float]:
    """Hard-coded monthly benchmark returns in percent."""
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

    # 1) Load data
    df, return_cols = load_ranked_returns("ranked_returns_top15.csv")

    # 2) Sidebar controls
    st.sidebar.header("Settings")
    num_positions = st.sidebar.slider("Number of Positions", 1, 15, 5)
    cash_pct = st.sidebar.slider("Cash %", 0.0, 100.0, 15.0) / 100.0

    # 3) Compute monthly portfolio returns
    portfolio_returns = compute_portfolio_returns(df, return_cols, num_positions, cash_pct)

    # 4) Compound the model portfolio
    model_series = (1 + portfolio_returns).cumprod() * INITIAL_INVESTMENT
    model_series.name = "Model Portfolio"

    # 5) Compound the benchmark
    bench_percents = get_actual_benchmark_returns()
    bench_returns = (
        pd.Series(bench_percents[: len(df)], index=df.index)
        .div(100)
    )
    actual_series = (1 + bench_returns).cumprod() * INITIAL_INVESTMENT
    actual_series.name = "Benchmark"

    # 6) Plot both
    combined = pd.concat([model_series, actual_series], axis=1)
    st.subheader("Portfolio Value Over Time")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(combined.index, combined["Model Portfolio"], label="Model Portfolio", linewidth=2)
    ax.plot(combined.index, combined["Benchmark"],       label="Benchmark (Actual)", linewidth=2, color="orange")
    ax.set_title("Model vs Benchmark Portfolio Value", fontsize=14)
    ax.set_xlabel("Date"); ax.set_ylabel("Value ($)")
    ax.legend(); ax.grid(True, linestyle="--", alpha=0.5)
    st.pyplot(fig)

    # 7) Performance metrics
    st.subheader("Model Performance")
    final_val     = model_series.iloc[-1]
    total_return  = (final_val - INITIAL_INVESTMENT) / INITIAL_INVESTMENT
    annual_return = (final_val / INITIAL_INVESTMENT) ** (12 / len(model_series)) - 1

    st.metric("Final Value",        f"${final_val:.2f}")
    st.metric("Total Return",       f"{total_return:.2%}")
    st.metric("Annualized Return",  f"{annual_return:.2%}")

    # 8) Optionally show the raw monthly returns
    with st.expander("Show Monthly Portfolio Returns"):
        st.dataframe(portfolio_returns.mul(100).round(2).astype(str) + "%")

if __name__ == "__main__":
    main()
