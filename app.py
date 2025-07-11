# app.py

import os
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Rank‐Weighted Backtest", layout="wide")

HERE = os.path.dirname(__file__)

@st.cache_data(show_spinner=False)
def load_returns():
    # build absolute path to the CSV
    path = os.path.join(HERE, "ranked_returns_top15.csv")
    df = pd.read_csv(path)

    # normalize & parse date
    df.columns = (
        df.columns
          .str.strip()
          .str.lower()
          .str.replace(r"\s+", "_", regex=True)
    )
    if "date" not in df.columns:
        raise KeyError(f"'date' column not found; saw {df.columns.tolist()}")
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()

    # identify return_rank columns
    ret_cols = sorted(
        [c for c in df.columns if c.startswith("return_rank_")],
        key=lambda c: int(c.rsplit("_",1)[-1])
    )

    # --- FIX: convert each column individually ---
    for c in ret_cols:
        df[c] = (
            df[c].astype(str)
                 .str.rstrip("%")
                 .astype(float)
        )

    return df, ret_cols

def compute_rank_weights(n, cash_pct):
    budget = 1.0 - cash_pct
    ranks = np.arange(n, 0, -1)
    return ranks / ranks.sum() * budget

def main():
    st.title("Rank‐Weighted Backtest")
    st.markdown("Monthly compounding • Linear rank weights • Custom rebalance interval")

    # 1) load
    df_ret, ret_cols = load_returns()

    # 2) sidebar
    N        = st.sidebar.slider("Number of positions (N)", 1, len(ret_cols), len(ret_cols))
    cash_pct = st.sidebar.slider("Cash % (uninvested)", 0.0, 100.0, 0.0) / 100.0
    reb_cost = st.sidebar.slider("Rebalance cost %", 0.0, 5.0, 0.1, step=0.01)/100.0
    freq     = st.sidebar.selectbox("Rebalance interval",
                                    ["Monthly","Quarterly","Semi-Annual"])
    interval = {"Monthly":1, "Quarterly":3, "Semi-Annual":6}[freq]

    weights = compute_rank_weights(N, cash_pct)
    st.sidebar.markdown("**Rank Weights**")
    for i,w in enumerate(weights, start=1):
        st.sidebar.write(f"Rank {i}: {w*100:.2f}%")
    st.sidebar.write(f"Cash: {cash_pct*100:.2f}%")
    st.sidebar.write(f"Rebalance every **{interval}** month(s)")

    # 3) build weight matrix
    df = df_ret.copy()
    periods = len(df)
    idxs = np.arange(periods)
    reb_flag = (idxs % interval == 0)

    w = pd.DataFrame(0.0, index=df.index, columns=ret_cols)
    w.iloc[reb_flag, :N] = weights
    w = w.ffill().fillna(0.0)

    # 4) compute period returns and compound
    period_ret = (w.values * (df[ret_cols].values / 100.0)).sum(axis=1)
    val = 100.0
    vals = []
    for r in period_ret:
        val *= (1 + r)
        vals.append(val)
    series = pd.Series(vals, index=df.index, name="Model Value")

    # 5) plot & metrics
    st.line_chart(series.to_frame())
    final = series.iloc[-1]
    total = final/100.0 - 1
    ann   = (final/100.0)**(12/periods) - 1

    c1,c2 = st.columns(2)
    c1.metric("Total Return",      f"{total:.2%}")
    c1.metric("Final Portfolio $", f"${final:,.2f}")
    c2.metric("Annualized Return", f"{ann:.2%}")
    c2.metric("Rebalance Cost %",  f"{reb_cost*100:.2f}%")

if __name__ == "__main__":
    main()
