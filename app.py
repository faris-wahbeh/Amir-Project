import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ── Constants ─────────────────────────────────────────────────────
INITIAL_INVESTMENT = 100.0     # starting portfolio value
DEFAULT_REBALANCE_MONTHS = 1   # fixed monthly rebalance

@st.cache_data(show_spinner=False)
def load_returns(returns_path: str):
    df = pd.read_csv(returns_path)
    df.columns = (
        df.columns
          .str.strip()
          .str.lower()
          .str.replace(r"\s+", "_", regex=True)
    )
    if 'date' not in df.columns:
        raise KeyError(f"Expected 'date' column but got {df.columns.tolist()}")
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').set_index('date')

    ret_cols = sorted(
        [c for c in df.columns if c.startswith('return_rank_')],
        key=lambda c: int(c.rsplit('_',1)[-1])
    )
    df[ret_cols] = df[ret_cols].replace('%','',regex=True).astype(float)
    return df, ret_cols

def compute_rank_weights(n: int, cash_pct: float) -> np.ndarray:
    if n < 1:
        return np.array([])
    budget = 1.0 - cash_pct
    ranks = np.arange(n, 0, -1)
    return ranks / ranks.sum() * budget

def backtest_value(
    df: pd.DataFrame,
    ret_cols: list[str],
    weights: np.ndarray
) -> pd.Series:
    periods = len(df)
    rebalance_flag = (np.arange(periods) % DEFAULT_REBALANCE_MONTHS == 0)

    w = pd.DataFrame(0.0, index=df.index, columns=ret_cols)
    w.loc[rebalance_flag, ret_cols[:len(weights)]] = weights
    w = w.ffill().fillna(0.0)

    period_ret = (w * df[ret_cols] / 100.0).sum(axis=1)
    value = (1 + period_ret).cumprod() * INITIAL_INVESTMENT
    return value.rename("Model Value")

def main():
    st.set_page_config(page_title="Rank‐Weighted Backtest", layout="wide")
    st.title("Rank‐Weighted Backtest")
    st.markdown("#### Fixed monthly rebalance & linear rank weights")

    # ── Load data & actual values ─────────────────────────────────
    df_ret, ret_cols = load_returns("ranked_returns_top15.csv")
    actual_values = [
        100.3754217,100.6093042,100.7130938,100.9294943,101.4759915,
        102.0228643,102.0853826,102.6042349,102.6261388,101.9537503,
        102.1926446,101.6121289,102.3295752,102.8702978,103.0677537,
        103.3805797,103.2973817,103.7454304,103.9352115,103.9605518,
        103.6064345,103.7279590,104.1375157,104.1756022,104.4792258,
        104.0444822,103.2254612,104.0623542,104.5925347,104.8517258,
        105.1694284,105.6111936,105.5117806,105.3648261,105.9828735,
        106.3996055,106.3143744,106.7251204,106.4707991,106.7052544,
        106.4543254,106.8157714,107.0669666,107.3284046,107.2540997,
        107.4856317,107.4339350,107.5401174,106.7673168,106.8153787,
        106.7807937,106.3149881,105.8487119,105.4077107,106.0940136,
        105.9652512,105.4391806,105.6791974,106.0533227
    ]
    actual_series = pd.Series(
        actual_values,
        index=df_ret.index[:len(actual_values)],
        name='Actual Value'
    )

    # ── Sidebar inputs ────────────────────────────────────────────
    st.sidebar.header("Portfolio Settings")
    n = st.sidebar.slider("Number of Positions", 1, len(ret_cols), len(ret_cols))
    cash_pct = st.sidebar.slider("Cash % (uninvested)", 0.0, 100.0, 15.0) / 100.0

    # compute and show rank weights
    weights = compute_rank_weights(n, cash_pct)
    st.sidebar.markdown("**Rank Weights**")
    for i, w in enumerate(weights, start=1):
        st.sidebar.write(f"Rank {i}: {w*100:.2f}%")
    st.sidebar.write(f"Cash: {cash_pct*100:.2f}%")

    # ── Run backtest ──────────────────────────────────────────────
    model_series = backtest_value(df_ret, ret_cols, weights)
    model_series = model_series.loc[actual_series.index]

    # ── Plot with Matplotlib ─────────────────────────────────────
    combined = pd.concat([model_series, actual_series], axis=1)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(combined.index, combined["Model Value"], label="Model Value", linewidth=2)
    ax.plot(combined.index, combined["Actual Value"], label="Actual Value", linewidth=2)
    ymin = combined.min().min() * 0.99
    ymax = combined.max().max() * 1.01
    ax.set_ylim(ymin, ymax)
    ax.set_title("Model vs Actual Portfolio Value", fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value ($)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    st.pyplot(fig)

    # ── Performance metrics ──────────────────────────────────────
    final_model  = model_series.iloc[-1]
    final_actual = actual_series.iloc[-1]
    total_model  = (final_model - INITIAL_INVESTMENT) / INITIAL_INVESTMENT
    total_actual = (final_actual - INITIAL_INVESTMENT) / INITIAL_INVESTMENT
    ann_model    = (final_model / INITIAL_INVESTMENT) ** (12/len(model_series)) - 1
    ann_actual   = (final_actual / INITIAL_INVESTMENT) ** (12/len(actual_series)) - 1

    st.subheader("Performance Metrics")
    c1, c2 = st.columns(2)
    c1.metric("Model Total Return", f"{total_model:.2%}")
    c1.metric("Model Final Value", f"${final_model:,.2f}")
    c1.metric("Model Annual Return", f"{ann_model:.2%}")
    c2.metric("Actual Total Return", f"{total_actual:.2%}")
    c2.metric("Actual Final Value", f"${final_actual:,.2f}")
    c2.metric("Actual Annual Return", f"{ann_actual:.2%}")

if __name__ == "__main__":
    main()
