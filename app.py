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
    # strip “%” and to float
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

# ──────── fixed compounding logic ───────────────────────────────────────────────
def calculate_monthly_portfolio_values(
    df: pd.DataFrame,
    return_cols: list[str],
    num_positions: int,
    cash_pct: float,
    initial_investment: float = INITIAL_INVESTMENT
) -> pd.Series:
    """
    Compounds portfolio value monthly using weighted returns.
    Each month: apply weights to fixed top-N return columns and compound previous value.
    """
    weights = calculate_declining_weights(num_positions, cash_pct)
    selected = return_cols[:num_positions]

    portfolio_values = []
    current_value = initial_investment

    for idx, row in df.iterrows():
        monthly_returns = row[selected].values / 100.0
        growth_factor = np.dot(weights, 1 + monthly_returns)
        current_value *= growth_factor
        portfolio_values.append((idx, current_value))

    return pd.Series(dict(portfolio_values), name="Portfolio Value")

# ──────── benchmark data ───────────────────────────────────────────────────────
def get_actual_benchmark_returns() -> list[float]:
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

    # ---- compute portfolio values ----
    portfolio_series = calculate_monthly_portfolio_values(
        df, return_cols, num_positions, cash_pct
    )

    # ---- show the monthly portfolio values ----
    st.subheader("Monthly Portfolio Value")
    st.dataframe(portfolio_series.round(2).to_frame("Value ($)"))

    # ---- benchmark compounding ----
    bench_pct = get_actual_benchmark_returns()[: len(df)]
    bench_factors = pd.Series(bench_pct, index=df.index).div(100).add(1.0)
    benchmark_series = bench_factors.cumprod().mul(INITIAL_INVESTMENT)
    benchmark_series.name = "Benchmark Value"

    # ---- plot both series ----
    combined = pd.concat([portfolio_series, benchmark_series], axis=1)
    st.subheader("Portfolio Value Over Time")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(combined.index, combined["Portfolio Value"], label="Model", linewidth=2)
    ax.plot(combined.index, combined["Benchmark Value"], label="Benchmark", color="orange", linewidth=2)
    ax.set_xlabel("Date")
    ax.set_ylabel("Value ($)")
    ax.set_title("Compounded Portfolio vs Benchmark")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
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
