import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ── Constants ─────────────────────────────────────────────────────
INITIAL_INVESTMENT = 100.0
DEFAULT_REBALANCE_MONTHS = 1

@st.cache_data(show_spinner=False)
def load_returns(path):
    df = pd.read_csv(path)
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", "_", regex=True)
    )
    if "date" not in df.columns:
        raise KeyError("Expected a 'date' column.")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")

    ret_cols = sorted(
        [c for c in df.columns if c.startswith("return_rank_")],
        key=lambda c: int(c.rsplit("_", 1)[-1])
    )
    df[ret_cols] = df[ret_cols].astype(str).replace("%", "", regex=True).astype(float)
    return df, ret_cols

@st.cache_data(show_spinner=False)
def load_names(path):
    df = pd.read_csv(path)
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", "_", regex=True)
    )
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")
    return df

def compute_rank_weights(n, cash_pct):
    if n < 1:
        return np.array([])
    budget = 1.0 - cash_pct
    ranks = np.arange(n, 0, -1)
    return ranks / ranks.sum() * budget

def main():
    st.set_page_config(page_title="Rank‐Weighted Backtest", layout="wide")
    st.title("Rank‐Weighted Backtest")
    st.markdown("#### Fixed monthly rebalance & linear rank weights")

    # ── Load CSVs ───────────────────────────────────────────────────
    df_ret, ret_cols = load_returns("ranked_returns_top15.csv")

    # Optional: Load ticker names (not used here)
    names_file = "ranked_by_exposure_top15.csv"
    if os.path.exists(names_file):
        df_names = load_names(names_file)
    else:
        df_names = None
        st.warning(f"'{names_file}' not found. Running without ticker names.")

    # ── Sidebar Inputs ─────────────────────────────────────────────
    st.sidebar.header("Portfolio Settings")
    n = st.sidebar.slider("Number of Positions", 1, len(ret_cols), len(ret_cols))
    cash_pct = st.sidebar.slider("Cash % (uninvested)", 0.0, 100.0, 15.0) / 100.0

    weights = compute_rank_weights(n, cash_pct)
    st.sidebar.markdown("**Rank Weights**")
    for i, w in enumerate(weights, start=1):
        st.sidebar.write(f"Rank {i}: {w*100:.2f}%")
    st.sidebar.write(f"Cash: {cash_pct*100:.2f}%")

    # ── Build weight matrix over time ──────────────────────────────
    periods = len(df_ret)
    rebalance_flag = (np.arange(periods) % DEFAULT_REBALANCE_MONTHS == 0)
    w = pd.DataFrame(0.0, index=df_ret.index, columns=ret_cols)
    w.loc[rebalance_flag, ret_cols[:n]] = weights
    w = w.ffill().fillna(0.0)

    # ── Compute portfolio monthly returns (weighted) ───────────────
    period_ret = (w * df_ret[ret_cols] / 100.0).sum(axis=1)

    # ── Compound returns ───────────────────────────────────────────
    model_values = [INITIAL_INVESTMENT]
    for r in period_ret:
        model_values.append(model_values[-1] * (1 + r))
    model_values = model_values[1:]
    model_series = pd.Series(
        data=model_values,
        index=period_ret.index,
        name="Model Value"
    )

    # ── Plot portfolio vs benchmark (display only) ─────────────────
    actual_returns = [
        5.34, 0.16, 1.4, 2.8, 4.98, 5.38, 1.27, 7.16, 0.81, -8.68,
        3.52, -8.29, 8.75, 9.03, 3.26, 5.04, -2.46, 6.89, 3.32, -0.27,
        -6.38, 0.6, 5.64, 0.68, 5.59, -6.47, -14.21, 14.33, 9.07, 3.96,
        6.06, 6.67, -3.01, -3.56, 11.92, 7.29, -2.44, 8.33, -6.72, 5.05,
        -6.17, 6.33, 3.65, 5.27, -2.18, 3.22, -1.75, 0.02, -12.82, -0.39,
        -2.01, -9.02, -9.27, -8.22, 9.11, -2.98, -7.97, 1.83, 4.05, -2.56
    ]
    actual_values = [INITIAL_INVESTMENT]
    for r in actual_returns:
        actual_values.append(actual_values[-1] * (1 + r / 100))
    actual_values = actual_values[1:]
    actual_series = pd.Series(
        data=actual_values,
        index=df_ret.index[:len(actual_values)],
        name="Actual Value"
    )

    combined = pd.concat([model_series, actual_series], axis=1)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(combined.index, combined["Model Value"], label="Model Value", linewidth=2)
    ax.plot(combined.index, combined["Actual Value"], label="Actual Value (Benchmark)", linewidth=2, color="orange")
    ax.set_title("Model vs Benchmark Portfolio Value", fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value ($)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    st.pyplot(fig)

    # ── Model-only performance metrics ─────────────────────────────
    final_m = model_series.iloc[-1]
    total_m = (final_m - INITIAL_INVESTMENT) / INITIAL_INVESTMENT
    ann_m = (final_m / INITIAL_INVESTMENT) ** (12 / len(model_series)) - 1

    st.subheader("Model Performance")
    st.metric("Total Return", f"{total_m:.2%}")
    st.metric("Final Value", f"${final_m:,.2f}")
    st.metric("Annual Return", f"{ann_m:.2%}")

    # ── Optional Debug Outputs ─────────────────────────────────────
    with st.expander("Debug: Monthly Returns and Weights"):
        st.dataframe(df_ret[ret_cols[:n]].head(12), use_container_width=True)
        st.dataframe(w[ret_cols[:n]].head(12), use_container_width=True)
        st.line_chart(period_ret.rename("Monthly Portfolio Return"))

if __name__ == "__main__":
    main()
