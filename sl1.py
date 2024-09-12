import streamlit as st
import pandas as pd
import numpy as np
import random

# Function to load and prepare data
@st.cache
def load_data(file):
    data = pd.read_csv(file)
    data['Datetime'] = pd.to_datetime(data['Datetime'])
    data.set_index('Datetime', inplace=True)
    return data

# Function to calculate indicators and perform backtest
def backtest(data, ema_period, rsi_period, macd_fast, macd_slow, macd_signal):
    # Calculate EMA
    data['EMA'] = data['Close'].ewm(span=ema_period, adjust=False).mean()

    # Calculate RSI
    delta = data['Close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    average_gain = gain.rolling(window=rsi_period, min_periods=1).mean()
    average_loss = loss.rolling(window=rsi_period, min_periods=1).mean()
    rs = average_gain / average_loss
    data['RSI'] = 100 - (100 / (1 + rs))

    # Calculate MACD
    fast_ema = data['Close'].ewm(span=macd_fast, adjust=False).mean()
    slow_ema = data['Close'].ewm(span=macd_slow, adjust=False).mean()
    data['MACD'] = fast_ema - slow_ema
    data['MACD_signal'] = data['MACD'].ewm(span=macd_signal, adjust=False).mean()
    data['MACD_hist'] = data['MACD'] - data['MACD_signal']

    # Generate signals
    data['signal'] = np.where(
        (data['Close'] > data['EMA']) & 
        (data['RSI'] > 50) & 
        (data['MACD'] > data['MACD_signal']), 1, 
        np.where(
            (data['Close'] < data['EMA']) & 
            (data['RSI'] < 50) & 
            (data['MACD'] < data['MACD_signal']), -1, 
            0
        )
    )
    
    # Calculate strategy returns
    data['strategy_returns'] = data['signal'].shift(1) * (data['Close'].pct_change())
    data['cumulative_returns'] = (1 + data['strategy_returns']).cumprod()

    return data['cumulative_returns'].iloc[-1]

# Define the Streamlit app
def main():
    st.title("Trading Strategy Optimization")

    # File upload
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        data = load_data(uploaded_file)
        st.write("Data Preview:", data.head())

        # Input fields for parameter ranges
        st.sidebar.subheader("Parameter Settings")
        n_samples = st.sidebar.slider("Number of Samples", 10, 1000, 100)

        if st.sidebar.button("Run Optimization"):
            # Define parameter ranges
            ema_range = range(10, 50, 2)
            rsi_range = range(7, 47, 2)
            macd_fast_range = range(8, 48, 2)
            macd_slow_range = range(17, 57, 2)
            macd_signal_range = range(7, 47, 2)

            # Randomly sample parameter combinations
            param_combinations = []
            for _ in range(n_samples):
                ema_period = random.choice(ema_range)
                rsi_period = random.choice(rsi_range)
                macd_fast = random.choice(macd_fast_range)
                macd_slow = random.choice(macd_slow_range)
                macd_signal = random.choice(macd_signal_range)

                if macd_fast < macd_slow:  # Ensure valid MACD settings
                    param_combinations.append((ema_period, rsi_period, macd_fast, macd_slow, macd_signal))

            # Perform backtest for each combination
            results = []
            for combo in param_combinations:
                ema_period, rsi_period, macd_fast, macd_slow, macd_signal = combo
                cumulative_return = backtest(data.copy(), ema_period, rsi_period, macd_fast, macd_slow, macd_signal)
                results.append({
                    'EMA_Period': ema_period,
                    'RSI_Period': rsi_period,
                    'MACD_Fast': macd_fast,
                    'MACD_Slow': macd_slow,
                    'MACD_Signal': macd_signal,
                    'Cumulative_Returns': cumulative_return
                })

            # Convert results to DataFrame
            results_df = pd.DataFrame(results)
            st.write("Top 5 Results:", results_df.sort_values(by='Cumulative_Returns', ascending=False).head(5))

            # Option to download results
            csv = results_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Results as CSV", data=csv, file_name='optimization_results.csv')

# Run the app
if __name__ == "__main__":
    main()
