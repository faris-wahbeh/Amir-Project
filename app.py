# app.py

import os
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Rank‐Weighted Backtest", layout="wide")

# ── Constants ─────────────────────────────────────────────────────
INITIAL_INVESTMENT = 100.0

# ── Hard‐coded “actual” monthly returns (in %) ────────────────────
# (Use your real actual returns here; must match the number of months in ranked_returns_top15.csv)
MONTHLY_RETURNS_ACTUAL = [
    0.32, 1.34, 0.52, 1.11, 0.36, -0.16, -0.60,  # Jan–Jul 2018 sample…
    # …continue for each month in your data
]

HERE = os.path.dirname(__file__)

# ── Data loader ───────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_returns():
    path = os.path.join(HERE, "ranked_returns_top15.csv")
    df = pd.read_csv(path)
    df.columns = (
        df.columns.str.strip()
                  .str.lower()
                  .str.replace(r"\s+", "_", regex=True)
    )
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()

    # find return_rank_X columns
    ret_cols = sorted(
        [c for c in df.columns if c.startswith("return_rank_")],
        key=lambda c: int(c.rsplit("_", 1)[-1])
    )
    # convert each one to float
    for c in ret_cols:
        df[c] = df[c].astype(str).str.rstrip("%").astype(float)
    return df, ret_cols

def compute_rank_weights(n, cash_pct):
    budget = 1.0 - cash_pct
    ranks = np.arange(n, 0, -1)
    return ranks / ranks.sum() * budget

def main():
    st.title("Rank‐Weighted Backtest vs Actual")
    st.markdown("Model and actual portfolio values, compounded monthly.")

    # 1) Load model returns
    df_ret, ret_cols = load_returns()
    dates = df_ret.index

    # 2) Sidebar
    N        = st.sidebar.slider("Number of positions (N)", 1, len(ret_cols), len(ret_cols))
    cash_pct = st.sidebar.slider("Cash % (uninvested)", 0.0, 100.0, 0.0) / 100.0
    reb_cost = st.sidebar.slider("Rebalance cost %", 0.0, 5.0, 0.1, step=0.01) / 100.0
    freq     = st.sidebar.selectbox("Rebalance frequency",
                                    ["Monthly","Quarterly","Semi-Annual"])
    interval = {"Monthly":1, "Quarterly":3, "Semi-Annual":6}[freq]

    weights = compute_rank_weights(N, cash_pct)
    st.sidebar.markdown("**Rank Weights**")
    for i,w in enumerate(weights, start=1):
        st.sidebar.write(f"Rank {i}: {w*100:.2f}%")
    st.sidebar.write(f"Cash: {cash_pct*100:.2f}%")
    st.sidebar.write(f"Rebalance every {interval} month(s)")

    # 3) Build weight matrix & compute model returns
    periods = len(df_ret)
    idxs    = np.arange(periods)
    reb_flag= (idxs % interval == 0)

    w = pd.DataFrame(0.0, index=dates, columns=ret_cols)
    w.iloc[reb_flag, :N] = weights
    w = w.ffill().fillna(0.0)

    # model period returns (decimal)
    period_ret = (w.values * (df_ret[ret_cols].values / 100.0)).sum(axis=1)

    # 4) Compound model series
    model_vals = [INITIAL_INVESTMENT]
    for r in period_ret:
        model_vals.append(model_vals[-1] * (1 + r))
    model_series = pd.Series(model_vals[1:], index=dates, name="Model Value")

    # 5) Build actual series (decimal returns from your list)
    if len(MONTHLY_RETURNS_ACTUAL) < periods:
        st.error(f"Need at least {periods} actual returns; got {len(MONTHLY_RETURNS_ACTUAL)}")
        return
    actual_vals = [INITIAL_INVESTMENT]
    for r in MONTHLY_RETURNS_ACTUAL[:periods]:
        actual_vals.append(actual_vals[-1] * (1 + r/100.0))
    actual_series = pd.Series(actual_vals[1:], index=dates, name="Actual Value")

    # 6) Plot both
    combined = pd.concat([model_series, actual_series], axis=1)
    st.line_chart(combined)

    # 7) Metrics
    final_m = model_series.iloc[-1]
    final_a = actual_series.iloc[-1]
    total_m = final_m/INITIAL_INVESTMENT - 1
    total_a = final_a/INITIAL_INVESTMENT - 1
    ann_m   = (final_m/INITIAL_INVESTMENT)**(12/periods) - 1
    ann_a   = (final_a/INITIAL_INVESTMENT)**(12/periods) - 1

    st.subheader("Performance Metrics")
    c1, c2 = st.columns(2)
    c1.metric("Model Total Return", f"{total_m:.2%}")
    c1.metric("Model Final Value", f"${final_m:,.2f}")
    c1.metric("Model Annual Return", f"{ann_m:.2%}")
    c2.metric("Actual Total Return", f"{total_a:.2%}")
    c2.metric("Actual Final Value", f"${final_a:,.2f}")
    c2.metric("Actual Annual Return", f"{ann_a:.2%}")

if __name__ == "__main__":
    main()
