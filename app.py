import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Constants
INITIAL_INVESTMENT = 100.0

@st.cache_data
def load_ranked_returns(file_path):
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
        [col for col in df.columns if col.startswith("return_rank_")],
        key=lambda c: int(c.split("_")[-1])
    )

    # Convert % strings to floats
    df[return_cols] = df[return_cols].replace("%", "", regex=True).astype(float)
    return df, return_cols

def calculate_declining_weights(num_positions, cash_pct):
    total_weight = 1.0 - cash_pct
    ranks = np.arange(num_positions, 0, -1)  # e.g., [5,4,3,2,1]
    weights = ranks / ranks.sum() * total_weight
    return weights

def get_actual_benchmark_returns():
    # Hardcoded actual monthly return %
    return [
        5.34, 0.16, 1.4, 2.8, 4.98, 5.38, 1.27, 7.16, 0.81, -8.68,
        3.52, -8.29, 8.75, 9.03, 3.26, 5.04, -2.46, 6.89, 3.32, -0.27,
        -6.38, 0.6, 5.64, 0.68, 5.59, -6.47, -14.21, 14.33, 9.07, 3.96,
        6.06, 6.67, -3.01, -3.56, 11.92, 7.29, -2.44, 8.33, -6.72, 5.05,
        -6.17, 6.33, 3.65, 5.27, -2.18, 3.22, -1.75, 0.02, -12.82, -0.39,
        -2.01, -9.02, -9.27, -8.22, 9.11, -2.98, -7.97, 1.83, 4.05, -2.56
    ]

def main():
    st.set_page_config(page_title="Compounding Weighted Portfolio", layout="wide")
    st.title("Ranked Portfolio vs Benchmark")

    # Load return data
    df, return_cols = load_ranked_returns("ranked_returns_top15.csv")

    # Sidebar inputs
    st.sidebar.header("Settings")
    num_positions = st.sidebar.slider("Number of Positions", min_value=1, max_value=15, value=5)
    cash_pct = st.sidebar.slider("Cash %", min_value=0.0, max_value=100.0, value=15.0) / 100.0

    weights = calculate_declining_weights(num_positions, cash_pct)

    # Use only top N rank columns
    selected_cols = return_cols[:num_positions]
    weighted_returns = df[selected_cols].div(100) * weights  # Elementwise return * weight

    # Sum weighted returns into a single portfolio return per month
    portfolio_returns = weighted_returns.sum(axis=1)

    # Compound model portfolio value
    model_values = [INITIAL_INVESTMENT]
    for r in portfolio_returns:
        model_values.append(model_values[-1] * (1 + r))
    model_values = model_values[1:]
    model_series = pd.Series(model_values, index=portfolio_returns.index, name="Model Portfolio")

    # Benchmark: hardcoded actual returns
    actual_returns = get_actual_benchmark_returns()
    actual_values = [INITIAL_INVESTMENT]
    for r in actual_returns[:len(df)]:
        actual_values.append(actual_values[-1] * (1 + r / 100))
    actual_values = actual_values[1:]
    actual_series = pd.Series(actual_values, index=df.index[:len(actual_values)], name="Benchmark")

    # Combine for plotting
    combined = pd.concat([model_series, actual_series], axis=1)

    # Plot
    st.subheader("Portfolio Value Over Time")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(combined.index, combined["Model Portfolio"], label="Model Portfolio", linewidth=2)
    ax.plot(combined.index, combined["Benchmark"], label="Benchmark (Actual)", linewidth=2, color="orange")
    ax.set_title("Model vs Benchmark Portfolio Value", fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Value ($)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    st.pyplot(fig)

    # Show performance metrics (Model only)
    st.subheader("Model Performance")
    final_value = model_series.iloc[-1]
    total_return = (final_value - INITIAL_INVESTMENT) / INITIAL_INVESTMENT
    annual_return = (final_value / INITIAL_INVESTMENT) ** (12 / len(model_series)) - 1

    st.metric("Final Value", f"${final_value:.2f}")
    st.metric("Total Return", f"{total_return:.2%}")
    st.metric("Annualized Return", f"{annual_return:.2%}")

    with st.expander("Show Monthly Portfolio Returns"):
        st.dataframe(portfolio_returns.apply(lambda x: f"{x*100:.2f}%"))

if __name__ == "__main__":
    main()
