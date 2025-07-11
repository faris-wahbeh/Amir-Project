import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- App Layout & Style ---
st.set_page_config(page_title="Top-N Portfolio Simulator", layout="centered")
st.title("📈 Top-N Weighted Portfolio Simulator")

st.markdown("""
This tool allows you to simulate a **Top-N ranked portfolio** using weighted returns. 
You can compare the result against a fixed reference portfolio.
""")

# --- 1. Portfolio Weighting Function ---
def calculate_weights(num_positions):
    total_weight = 1.0
    if num_positions <= 5:
        top_weight = 0.3 * total_weight
    else:
        top_weight = 0.3 * total_weight - (num_positions - 5) * 0.03 * total_weight

    a = top_weight
    n = num_positions
    S = total_weight
    d = (2 * (a * n) - 2 * S) / (n * (n - 1)) if n > 1 else 0
    weights = [a - i * d for i in range(n)]
    return weights

# --- 2. Portfolio Return Calculation ---
def calculate_portfolio_returns(df, num_positions):
    weights = calculate_weights(num_positions)
    monthly_returns = []

    for idx, row in df.iterrows():
        returns = row.iloc[:num_positions].str.rstrip('%').astype(float) / 100
        weighted_return = np.dot(weights, returns)
        monthly_returns.append(weighted_return)

    return pd.Series(monthly_returns, index=df.index)

# --- 3. Compounding Returns ---
def compound_returns(returns, initial=100):
    compounded = [initial]
    for r in returns:
        compounded.append(compounded[-1] * (1 + r))
    return compounded[1:]

# --- 4. Load Data from Local CSV ---
@st.cache_data
def load_data():
    return pd.read_csv("ranked_returns_top15.csv", index_col=0, parse_dates=True, dayfirst=True)

# --- Run App Logic ---
df = load_data()
num_positions = st.slider("🎯 Select number of top positions", min_value=1, max_value=15, value=5)

# Simulated portfolio
portfolio_returns = calculate_portfolio_returns(df, num_positions)
portfolio_compounded = compound_returns(portfolio_returns)

# Reference/base portfolio
base_returns = [ 
    5.34, 0.16, 1.4, 2.8, 4.98, 5.38, 1.27, 7.16, 0.81, -8.68,
    3.52, -8.29, 8.75, 9.03, 3.26, 5.04, -2.46, 6.89, 3.32, -0.27,
    -6.38, 0.6, 5.64, 0.68, 5.59, -6.47, -14.21, 14.33, 9.07, 3.96,
    6.06, 6.67, -3.01, -3.56, 11.92, 7.29, -2.44, 8.33, -6.72, 5.05,
    -6.17, 6.33, 3.65, 5.27, -2.18, 3.22, -1.75, 0.02, -12.82, -0.39,
    -2.01, -9.02, -9.27, -8.22, 9.11, -2.98, -7.97, 1.83, 4.05, -2.56
]

base_returns_series = pd.Series(base_returns[:len(df)]) / 100
base_compounded = compound_returns(base_returns_series)

# --- Plotting Section ---
st.subheader("📊 Portfolio Performance Comparison")

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(df.index[:len(portfolio_compounded)], portfolio_compounded, label="Top-N Portfolio", linewidth=2)
ax.plot(df.index[:len(base_compounded)], base_compounded, label="Reference Portfolio", linestyle='--', linewidth=2)

ax.set_title("Portfolio Growth Over Time", fontsize=14)
ax.set_ylabel("Portfolio Value", fontsize=12)
ax.set_xlabel("Date", fontsize=12)
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend()
plt.xticks(rotation=45)
st.pyplot(fig)

# --- Return Table ---
st.subheader("🧮 Monthly Portfolio Returns")
returns_display = portfolio_returns.map(lambda x: f"{x*100:.2f}%")
st.dataframe(returns_display.rename("Return %"))

# Optional Note
st.markdown("ℹ️ This simulation uses a decreasing linear weighting scheme. You can expand this with rebalancing, costs, or volatility targeting.")
