import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

# --- 4. Load Data ---
@st.cache_data
def load_data(file):
    df = pd.read_csv(file, index_col=0, parse_dates=True, dayfirst=True)
    return df

# --- Streamlit UI ---
st.title("Weighted Top-N Portfolio Simulator")

uploaded_file = st.file_uploader("Upload `ranked_returns_top15.csv`", type="csv")

if uploaded_file:
    df = load_data(uploaded_file)
    num_positions = st.slider("Number of positions to invest in:", min_value=1, max_value=15, value=5)

    # Portfolio computation
    portfolio_returns = calculate_portfolio_returns(df, num_positions)
    portfolio_compounded = compound_returns(portfolio_returns)

    # Base portfolio (from your list)
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

    # Plotting
    fig, ax = plt.subplots()
    ax.plot(df.index[:len(portfolio_compounded)], portfolio_compounded, label="Top-N Portfolio")
    ax.plot(df.index[:len(base_compounded)], base_compounded, label="Base Portfolio", linestyle='--')
    ax.set_ylabel("Portfolio Value")
    ax.set_title("Portfolio Growth Over Time")
    ax.legend()
    st.pyplot(fig)

    # Show returns
    st.subheader("Monthly Portfolio Returns")
    st.dataframe(portfolio_returns.apply(lambda x: f"{x*100:.2f}%"))

