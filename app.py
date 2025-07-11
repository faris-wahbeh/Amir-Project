import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ── Constants ─────────────────────────────────────────────────────
INITIAL_INVESTMENT = 100.0     # starting portfolio value
DEFAULT_REBALANCE_MONTHS = 1   # fixed monthly rebalance

@st.cache_data(show_spinner=False)
def load_returns(path: str):
    """
    Load rank‐based returns CSV, clean column names, parse dates,
    strip '%' and convert to floats (e.g. 5.34 → 0.0534).
    Returns DataFrame indexed by date plus list of ret_cols.
    """
    df = pd.read_csv(path)
    # normalize column names
    df.columns = (
        df.columns
          .str.strip()
          .str.lower()
          .str.replace(r"\s+", "_", regex=True)
    )
    if "date" not in df.columns:
        raise KeyError(f"Expected a 'date' column, got {df.columns.tolist()}")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")

    # detect return columns
    ret_cols = sorted(
        [c for c in df.columns if c.startswith("return_rank_")],
        key=lambda c: int(c.rsplit("_",1)[-1])
    )
    # strip percent signs, convert to float, then to decimal
    df[ret_cols] = (
        df[ret_cols]
          .replace("%","", regex=True)
          .astype(float)
          .div(100.0)
    )
    return df, ret_cols

def compute_rank_weights(n: int, cash_pct: float) -> np.ndarray:
    """
    Linear declining weights for ranks 1…n summing to (1 - cash_pct).
    """
    if n < 1:
        return np.array([])
    budget = 1.0 - cash_pct
    ranks = np.arange(n, 0, -1)
    return ranks / ranks.sum() * budget

def main():
    st.set_page_config(page_title="Rank‐Weighted Backtest", layout="wide")
    st.title("Rank‐Weighted Backtest")
    st.markdown("#### Fixed monthly rebalance & linear rank weights")

    # ── Load model return data ─────────────────────────────────────
    df_ret, ret_cols = load_returns("ranked_returns_top15.csv")

    # ── User‐provided actual monthly RETURN %s ────────────────────
    monthly_returns = [
        5.34, 0.16, 1.4, 2.8, 4.98, 5.38, 1.27, 7.16, 0.81, -8.68,
        3.52, -8.29, 8.75, 9.03, 3.26, 5.04, -2.46, 6.89, 3.32, -0.27,
        -6.38, 0.6, 5.64, 0.68, 5.59, -6.47, -14.21, 14.33, 9.07, 3.96,
        6.06, 6.67, -3.01, -3.56, 11.92, 7.29, -2.44, 8.33, -6.72, 5.05,
        -6.17, 6.33, 3.65, 5.27, -2.18, 3.22, -1.75, 0.02, -12.82, -0.39,
        -2.01, -9.02, -9.27, -8.22, 9.11, -2.98, -7.97, 1.83, 4.05, -2.56
    ]
    # compound actual returns
    actual_series = (
        pd.Series(monthly_returns, index=df_ret.index[:len(monthly_returns)])
          .add(1)
          .cumprod()
          .mul(INITIAL_INVESTMENT)
          .rename("Actual Value")
    )

    # ── Sidebar: portfolio inputs ──────────────────────────────────
    st.sidebar.header("Portfolio Settings")
    n = st.sidebar.slider("Number of Positions", 1, len(ret_cols), len(ret_cols))
    cash_pct = st.sidebar.slider("Cash % (uninvested)", 0.0, 100.0, 15.0) / 100.0

    # compute and display rank‐based weights
    weights = compute_rank_weights(n, cash_pct)
    st.sidebar.markdown("**Rank Weights**")
    for i, w in enumerate(weights, start=1):
        st.sidebar.write(f"Rank {i}: {w*100:.2f}%")
    st.sidebar.write(f"Cash: {cash_pct*100:.2f}%")

    # ── Model: compute period returns ──────────────────────────────
    periods = len(df_ret)
    rebalance_flag = (np.arange(periods) % DEFAULT_REBALANCE_MONTHS == 0)
    w = pd.DataFrame(0.0, index=df_ret.index, columns=ret_cols)
    w.loc[rebalance_flag, ret_cols[:len(weights)]] = weights
    w = w.ffill().fillna(0.0)

    # weighted sum of decimal returns
    period_ret = (w * df_ret[ret_cols]).sum(axis=1)

    # ── Model: compound returns via cumprod ───────────────────────
    model_series = (1 + period_ret).cumprod().mul(INITIAL_INVESTMENT)
    model_series.name = "Model Value"

    # ── Plot both series with zoomed y‐axis ───────────────────────
    combined = pd.concat([model_series, actual_series], axis=1)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(combined.index, combined["Model Value"],  label="Model Value", linewidth=2)
    ax.plot(combined.index, combined["Actual Value"], label="Actual Value", linewidth=2)
    ymin, ymax = combined.min().min() * 0.99, combined.max().max() * 1.01
    ax.set_ylim(ymin, ymax)
    ax.set_title("Model vs Actual Portfolio Value", fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value ($)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    st.pyplot(fig)

    # ── Performance metrics ────────────────────────────────────────
    final_m = model_series.iloc[-1]
    final_a = actual_series.iloc[-1]
    total_m = (final_m - INITIAL_INVESTMENT) / INITIAL_INVESTMENT
    total_a = (final_a - INITIAL_INVESTMENT) / INITIAL_INVESTMENT
    ann_m   = (final_m / INITIAL_INVESTMENT) ** (12/len(model_series)) - 1
    ann_a   = (final_a / INITIAL_INVESTMENT) ** (12/len(actual_series)) - 1

    st.subheader("Performance Metrics")
    c1, c2 = st.columns(2)
    c1.metric("Model Total Return",  f"{total_m:.2%}")
    c1.metric("Model Final Value",  f"${final_m:,.2f}")
    c1.metric("Model Annual Return", f"{ann_m:.2%}")
    c2.metric("Actual Total Return",  f"{total_a:.2%}")
    c2.metric("Actual Final Value",  f"${final_a:,.2f}")
    c2.metric("Actual Annual Return", f"{ann_a:.2%}")

if __name__ == "__main__":
    main()
