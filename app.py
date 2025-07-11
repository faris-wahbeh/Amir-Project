import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ── Constants ─────────────────────────────────────────────────────
INITIAL_INVESTMENT = 100.0
DEFAULT_REBALANCE_MONTHS = 1   # fixed monthly rebalance

@st.cache_data(show_spinner=False)
def load_data(exp_file, ret_file):
    # Read Excel sheets
    df_exp = pd.read_excel(exp_file)
    df_ret = pd.read_excel(ret_file)

    # Standardize cols and parse dates
    def normalize(df):
        df.columns = (
            df.columns
              .str.strip()
              .str.lower()
              .str.replace(r"\s+", "_", regex=True)
        )
        if "date" not in df.columns:
            raise KeyError(f"'date' column missing; saw {df.columns.tolist()}")
        df["date"] = pd.to_datetime(df["date"])
        return df.sort_values("date").reset_index(drop=True)

    df_exp = normalize(df_exp)
    df_ret = normalize(df_ret)

    # Identify return‐rank columns
    ret_cols = sorted(
        [c for c in df_ret.columns if c.startswith("return_rank_")],
        key=lambda c: int(c.rsplit("_",1)[-1])
    )
    # Strip "%" and convert to float
    df_ret[ret_cols] = (
        df_ret[ret_cols]
          .astype(str)
          .replace("%","",regex=True)
          .astype(float)
    )

    return df_exp, df_ret, ret_cols

def compute_rank_weights(n, cash_pct):
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
    st.markdown("#### Upload your Exposure & Returns Excel, choose N & Cash%, and see the compounded \$100 curve")

    # ── File upload ────────────────────────────────────────────────
    exp_file = st.sidebar.file_uploader("Exposure Excel file", type=["xlsx"])
    ret_file = st.sidebar.file_uploader("Returns Excel file",  type=["xlsx"])
    if not exp_file or not ret_file:
        st.sidebar.info("Upload both Excel files to proceed")
        return

    # ── Load & prep data ───────────────────────────────────────────
    df_exp, df_ret, ret_cols = load_data(exp_file, ret_file)

    # ── Sidebar inputs ─────────────────────────────────────────────
    st.sidebar.header("Portfolio Settings")
    n        = st.sidebar.slider("Number of Positions (N)", 1, len(ret_cols), len(ret_cols))
    cash_pct = st.sidebar.slider("Cash % (uninvested)", 0.0, 100.0, 15.0) / 100.0

    # Show computed linear rank‐weights
    weights = compute_rank_weights(n, cash_pct)
    st.sidebar.markdown("**Rank Weights (summing to 100 – cash%)**")
    for i, w in enumerate(weights, start=1):
        st.sidebar.write(f"Rank {i}: {w*100:.2f}%")
    st.sidebar.write(f"Cash: {cash_pct*100:.2f}%")

    # ── Build monthly portfolio and period‐returns ────────────────
    dates          = df_ret["date"] if "date" in df_ret.columns else df_ret.index
    periods        = len(df_ret)
    rebalance_flag = (np.arange(periods) % DEFAULT_REBALANCE_MONTHS == 0)
    W              = pd.DataFrame(0.0, index=df_ret.index, columns=ret_cols)

    # On each rebalance date, apply the top‐N weights
    W.loc[rebalance_flag, ret_cols[:n]] = weights
    W = W.ffill().fillna(0.0)

    # Compute period‐returns as sum(weight_i * return_i)
    period_ret     = (W * df_ret[ret_cols] / 100.0).sum(axis=1)
    model_returns  = period_ret.tolist()

    # ── Compound the model from $100 ───────────────────────────────
    model_values = [INITIAL_INVESTMENT]
    for r in model_returns:
        model_values.append(model_values[-1] * (1 + r))
    model_values = model_values[1:]
    model_series = pd.Series(model_values, index=df_ret.index, name="Model Value")

    # ── If you have an “Actual Returns” sheet, do the same: ───────
    # (Otherwise skip this section; you can visualize just the model)
    # actual_df = pd.read_excel(exp_file, sheet_name="ActualReturns")  # example
    # … parse & compound …

    # ── Plot ───────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(model_series.index, model_series.values, label="Model Value", linewidth=2)
    ax.set_title("Compounded Portfolio Value (Start = $100)", fontsize=14)
    ax.set_xlabel("Date"); ax.set_ylabel("Portfolio Value ($)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    st.pyplot(fig)

    # ── Performance Metrics ────────────────────────────────────────
    final_val = model_series.iloc[-1]
    total_ret = (final_val - INITIAL_INVESTMENT)/INITIAL_INVESTMENT
    ann_ret   = (final_val/INITIAL_INVESTMENT)**(12/len(model_series)) - 1

    st.subheader("Performance Metrics")
    c1, c2 = st.columns(2)
    c1.metric("Final Value", f"${final_val:,.2f}")
    c1.metric("Total Return", f"{total_ret:.2%}")
    c2.metric("Annualized Return", f"{ann_ret:.2%}")

if __name__ == "__main__":
    main()
