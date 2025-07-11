# app.py

import streamlit as st
import pandas as pd
import numpy as np

# ── Constants ─────────────────────────────────────────────────────
INITIAL_INVESTMENT = 100.0

# ── Data loading ──────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_returns(path="ranked_returns_top15.csv"):
    df = pd.read_csv(path)
    # normalize & parse date
    df.columns = (
        df.columns
          .str.strip()
          .str.lower()
          .str.replace(r"\s+", "_", regex=True)
    )
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")

    # find the return_rank_X columns
    ret_cols = sorted(
        [c for c in df.columns if c.startswith("return_rank_")],
        key=lambda c: int(c.rsplit("_",1)[-1])
    )
    # strip "%" and to float
    df[ret_cols] = df[ret_cols].astype(str).str.rstrip("%").astype(float)
    return df, ret_cols

# (Optional) if you ever need exposures:
# @st.cache_data(show_spinner=False)
# def load_exposures(path="ranked_by_exposure_top15.csv"):
#     df = pd.read_csv(path)
#     ... normalize + parse date + strip "%"... 
#     return df, exp_cols

# ── Weighting utility ─────────────────────────────────────────────
def compute_rank_weights(n, cash_pct):
    """
    Linear declining weights for ranks 1…n that sum to (1 - cash_pct).
    """
    budget = 1.0 - cash_pct
    ranks = np.arange(n, 0, -1)
    return ranks / ranks.sum() * budget


# ── Streamlit App ─────────────────────────────────────────────────
def main():
    st.set_page_config(page_title="Rank-Weighted Backtest", layout="wide")
    st.title("Rank-Weighted Backtest")
    st.markdown("Monthly compounding • Linear rank weights • Custom rebalance frequency")

    # 1) Load returns pivot
    df_ret, ret_cols = load_returns()

    # 2) Sidebar controls
    st.sidebar.header("Portfolio Settings")
    N = st.sidebar.slider("Number of positions (N)", 1, len(ret_cols), len(ret_cols))
    cash_pct = st.sidebar.slider("Cash % (uninvested)", 0.0, 100.0, 0.0) / 100.0
    reb_cost = st.sidebar.slider("Rebalance cost %", 0.0, 5.0, 0.1, step=0.01) / 100.0
    freq = st.sidebar.selectbox("Rebalance frequency",
                                ["Monthly", "Quarterly", "Semi-Annual"])

    # map frequency to interval
    interval_map = {"Monthly": 1, "Quarterly": 3, "Semi-Annual": 6}
    interval = interval_map[freq]

    # 3) Compute the rank weights
    weights = compute_rank_weights(N, cash_pct)
    st.sidebar.markdown("**Rank Weights**")
    for i, w in enumerate(weights, start=1):
        st.sidebar.write(f"Rank {i}: {w*100:.2f}%")
    st.sidebar.write(f"Cash: {cash_pct*100:.2f}%")
    st.sidebar.write(f"Rebalance every **{interval}** month(s)")

    # 4) Build the weight matrix over time
    df = df_ret.copy()
    periods = len(df)
    idxs = np.arange(periods)

    # flag rows to rebalance on
    rebalance_flag = (idxs % interval == 0)

    # DataFrame of zeros, then set weights on rebalance points
    w = pd.DataFrame(0.0, index=df.index, columns=ret_cols)
    w.iloc[rebalance_flag, :N] = weights  # first N columns get the weights
    w = w.ffill().fillna(0.0)

    # 5) Compute period returns (decimal): df_ret holds percents, so divide by 100
    period_ret = (w.values * (df[ret_cols].values / 100.0)).sum(axis=1)

    # 6) Compound the portfolio value
    model_values = [INITIAL_INVESTMENT]
    for r in period_ret:
        model_values.append(model_values[-1] * (1 + r))
    model_series = pd.Series(
        data=model_values[1:],  # drop the initial seed
        index=df.index,
        name="Model Value"
    )

    # 7) Plot
    st.line_chart(model_series.to_frame())

    # 8) Metrics
    final_val = model_series.iloc[-1]
    total_ret = final_val / INITIAL_INVESTMENT - 1
    ann_ret = (final_val / INITIAL_INVESTMENT) ** (12/periods) - 1

    st.subheader("Performance Metrics")
    col1, col2 = st.columns(2)
    col1.metric("Total Return", f"{total_ret:.2%}")
    col1.metric("Final Value", f"${final_val:,.2f}")
    col2.metric("Annualized Return", f"{ann_ret:.2%}")
    col2.metric("Rebalance Cost", f"{reb_cost*100:.2f}% per event")


if __name__ == "__main__":
    main()
