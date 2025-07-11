import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ── Constants ─────────────────────────────────────────────────────
INITIAL_INVESTMENT       = 100.0   # starting portfolio value
DEFAULT_REBALANCE_MONTHS = 1       # fixed monthly rebalance

@st.cache_data(show_spinner=False)
def load_data(exp_path: str, ret_path: str):
    """
    Load exposures & returns from CSV, clean column names & dates,
    identify the return_rank_* columns, strip '%' and convert to float.
    """
    df_exp = pd.read_csv(exp_path)
    df_ret = pd.read_csv(ret_path)

    def normalize(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
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

    ret_cols = sorted(
        [c for c in df_ret.columns if c.startswith("return_rank_")],
        key=lambda c: int(c.rsplit("_",1)[-1])
    )

    # strip "%" if present and convert to float
    df_ret[ret_cols] = (
        df_ret[ret_cols]
          .astype(str)
          .str.rstrip("%")
          .astype(float)
    )

    return df_exp, df_ret, ret_cols

def compute_rank_weights(n: int, cash_pct: float) -> np.ndarray:
    """
    Linear declining weights for ranks 1…n summing to (1 - cash_pct).
    """
    if n < 1:
        return np.array([])
    budget = 1.0 - cash_pct
    ranks  = np.arange(n, 0, -1)
    return ranks / ranks.sum() * budget

def main():
    st.set_page_config(page_title="Rank‐Weighted Backtest", layout="wide")
    st.title("Rank‐Weighted Backtest")
    st.markdown("#### Fixed monthly rebalance & linear rank weights")

    # ── Load CSVs from same folder ─────────────────────────────────
    try:
        df_exp, df_ret, ret_cols = load_data(
            "ranked_by_exposure_top15.csv",
            "ranked_returns_top15.csv"
        )
    except FileNotFoundError as e:
        st.error(f"CSV not found: {e.filename}")
        return
    except KeyError as e:
        st.error(str(e))
        return

    # ── Sidebar: portfolio inputs ──────────────────────────────────
    st.sidebar.header("Portfolio Settings")
    n        = st.sidebar.slider("Number of Positions (N)", 1, len(ret_cols), len(ret_cols))
    cash_pct = st.sidebar.slider("Cash % (uninvested)", 0.0, 100.0, 15.0) / 100.0

    # compute & display linear rank‐weights
    weights = compute_rank_weights(n, cash_pct)
    st.sidebar.markdown("**Rank Weights (sum to 100% − cash)**")
    for i, w in enumerate(weights, start=1):
        st.sidebar.write(f"Rank {i}: {w*100:.2f}%")
    st.sidebar.write(f"Cash: {cash_pct*100:.2f}%")

    # ── Build weight matrix & compute model returns ───────────────
    periods        = len(df_ret)
    rebalance_flag = (np.arange(periods) % DEFAULT_REBALANCE_MONTHS == 0)

    W = pd.DataFrame(0.0, index=df_ret.index, columns=ret_cols)
    W.loc[rebalance_flag, ret_cols[:n]] = weights
    W = W.ffill().fillna(0.0)

    period_ret    = (W * df_ret[ret_cols] / 100.0).sum(axis=1)
    model_returns = period_ret.tolist()

    # ── Compound model returns explicitly ─────────────────────────
    model_values = [INITIAL_INVESTMENT]
    for r in model_returns:
        model_values.append(model_values[-1] * (1 + r))
    model_values = model_values[1:]
    model_series = pd.Series(model_values, index=df_ret.index, name="Model Value")

    # ── Define & compound actual returns ──────────────────────────
    monthly_returns = [
        5.34, 0.16, 1.4, 2.8, 4.98, 5.38, 1.27, 7.16, 0.81, -8.68,
        3.52, -8.29, 8.75, 9.03, 3.26, 5.04, -2.46, 6.89, 3.32, -0.27,
        -6.38, 0.6, 5.64, 0.68, 5.59, -6.47, -14.21, 14.33, 9.07, 3.96,
        6.06, 6.67, -3.01, -3.56, 11.92, 7.29, -2.44, 8.33, -6.72, 5.05,
        -6.17, 6.33, 3.65, 5.27, -2.18, 3.22, -1.75, 0.02, -12.82, -0.39,
        -2.01, -9.02, -9.27, -8.22, 9.11, -2.98, -7.97, 1.83, 4.05, -2.56
    ]
    actual_values = [INITIAL_INVESTMENT]
    for r in monthly_returns:
        actual_values.append(actual_values[-1] * (1 + r/100))
    actual_values = actual_values[1:]
    # align actual_series to model index length
    actual_series = pd.Series(
        data=actual_values,
        index=df_ret.index[:len(actual_values)],
        name="Actual Value"
    )
    # trim model_series to actual length
    model_series = model_series.iloc[:len(actual_series)]

    # ── Plot both curves ───────────────────────────────────────────
    combined = pd.concat([model_series, actual_series], axis=1)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(combined.index, combined["Model Value"],  label="Model Value", linewidth=2)
    ax.plot(combined.index, combined["Actual Value"], label="Actual Value", linewidth=2)
    ymin, ymax = combined.min().min()*0.99, combined.max().max()*1.01
    ax.set_ylim(ymin, ymax)
    ax.set_title("Model vs Actual Portfolio Value (Starting $100)", fontsize=14)
    ax.set_xlabel("Date"); ax.set_ylabel("Portfolio Value ($)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    st.pyplot(fig)

    # ── Performance Metrics ────────────────────────────────────────
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
