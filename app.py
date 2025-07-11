import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ──────── constants ────────────────────────────────────────────────────────────
INITIAL_INVESTMENT = 100.0

# ──────── data loading & cleaning ───────────────────────────────────────────────
@st.cache_data
def load_ranked_returns(file_path: str):
    df = pd.read_csv(file_path)
    # normalize column names
    df.columns = (
        df.columns
          .str.strip()
          .str.lower()
          .str.replace(r"\s+", "_", regex=True)
    )
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")

    # pick out the return_rank_X columns in order
    return_cols = sorted(
        [c for c in df.columns if c.startswith("return_rank_")],
        key=lambda c: int(c.split("_")[-1])
    )
    # "1.23%" → 1.23 (float)
    df[return_cols] = df[return_cols].replace("%", "", regex=True).astype(float)
    return df, return_cols

# ──────── weight calculation ────────────────────────────────────────────────────
def calculate_declining_weights(num_positions: int, cash_pct: float) -> np.ndarray:
    """
    Linear declining weights from rank=1…N that sum to (1 - cash_pct).
    """
    total = 1.0 - cash_pct
    ranks = np.arange(num_positions, 0, -1)  # [N, N-1, …, 1]
    return ranks / ranks.sum() * total

# ──────── per‐column compounding ────────────────────────────────────────────────
def compute_position_values(
    df: pd.DataFrame,
    return_cols: list[str],
    num_positions: int,
    cash_pct: float,
    initial_investment: float = INITIAL_INVESTMENT
) -> pd.DataFrame:
    """
    Returns a DataFrame (same index as df, columns=top‐N ranks) where
      value_t,i = allocation_i * ∏_{s=0..t} (1 + return_s,i/100).
    """
    weights   = calculate_declining_weights(num_positions, cash_pct)
    selected  = return_cols[:num_positions]
    # dollars allocated to each rank-column
    allocation = weights * initial_investment

    # build the factor series for each column: 1 + return/100
    factors = df[selected].div(100).add(1.0)
    # cumulative product down each column
    cum_factors = factors.cumprod()
    # multiply by the allocation dollars
    position_values = cum_factors.multiply(allocation, axis=1)
    position_values.columns = selected
    return position_values

# ──────── benchmark data ───────────────────────────────────────────────────────
def get_actual_benchmark_returns() -> list[float]:
    """Hard‐coded actual monthly return % for the benchmark."""
    return [
        5.34, 0.16, 1.4, 2.8, 4.98, 5.38, 1.27, 7.16, 0.81, -8.68,
        3.52, -8.29, 8.75, 9.03, 3.26, 5.04, -2.46, 6.89, 3.32, -0.27,
        -6.38, 0.6, 5.64, 0.68, 5.59, -6.47, -14.21, 14.33, 9.07, 3.96,
        6.06, 6.67, -3.01, -3.56, 11.92, 7.29, -2.44, 8.33, -6.72, 5.05,
        -6.17, 6.33, 3.65, 5.27, -2.18, 3.22, -1.75, 0.02, -12.82, -0.39,
        -2.01, -9.02, -9.27, -8.22, 9.11, -2.98, -7.97, 1.83, 4.05, -2.56
    ]

# ──────── app ─────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(page_title="Compounded Weighted Portfolio", layout="wide")
    st.title("Ranked Portfolio vs Benchmark")

    # ---- load data ----
    df, return_cols = load_ranked_returns("ranked_returns_top15.csv")

    # ---- user inputs ----
    st.sidebar.header("Settings")
    num_positions = st.sidebar.slider("Number of Positions", 1, 15, 5)
    cash_pct      = st.sidebar.slider("Cash %", 0.0, 100.0, 15.0) / 100.0

    # ---- compute compounded position values ----
    pos_vals = compute_position_values(df, return_cols, num_positions, cash_pct)

    # ---- portfolio total value each month ----
    portfolio_series = pos_vals.sum(axis=1)
    portfolio_series.name = "Model Portfolio Value"

    # ---- derive monthly net returns from the compounded values ----
    monthly_net_return = portfolio_series.pct_change().fillna(0)

    # ---- show the position‐by‐position values ----
    with st.expander("📊 Position Values Over Time"):
        st.dataframe(pos_vals.style.format("{:.2f}"))

    # ---- show the monthly net returns ----
    st.subheader("Monthly Net Returns")
    st.dataframe(
        (monthly_net_return * 100)
          .round(2)
          .astype(str)
          .add("%")
          .to_frame("Net Return")
    )

    # ---- compound the benchmark the same way ----
    bench_pct = get_actual_benchmark_returns()[: len(df)]
    bench_factors = pd.Series(bench_pct, index=df.index).div(100).add(1.0)
    benchmark_series = bench_factors.cumprod().mul(INITIAL_INVESTMENT)
    benchmark_series.name = "Benchmark Value"

    # ---- plot both ----
    combined = pd.concat([portfolio_series, benchmark_series], axis=1)
    st.subheader("Portfolio Value Over Time")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(combined.index, combined["Model Portfolio Value"], label="Model", linewidth=2)
    ax.plot(combined.index, combined["Benchmark Value"],       label="Benchmark", color="orange", linewidth=2)
    ax.set_xlabel("Date")
    ax.set_ylabel("Value ($)")
    ax.set_title("Compounded Portfolio vs Benchmark")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    st.pyplot(fig)

    # ---- performance metrics ----
    st.subheader("Model Performance")
    final_val = portfolio_series.iloc[-1]
    total_ret = (final_val - INITIAL_INVESTMENT) / INITIAL_INVESTMENT
    ann_ret   = (final_val / INITIAL_INVESTMENT) ** (12 / len(portfolio_series)) - 1

    st.metric("Final Value",       f"${final_val:.2f}")
    st.metric("Total Return",      f"{total_ret:.2%}")
    st.metric("Annualized Return", f"{ann_ret:.2%}")

if __name__ == "__main__":
    main()
