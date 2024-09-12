import csv
from SmartApi import SmartConnect
import os
import urllib
import json
import pandas as pd
import datetime as dt
import time
from pyotp import TOTP
from st2 import apply_pivots, custom_resample
import numpy as np
import math
from login_manager import LoginManager
from instruments_list import InstrumentDetail
import pandas_ta as ta
lm = LoginManager()
starttime = time.time()


obj, data, usr = lm.login_and_retrieve_info()

il = InstrumentDetail()
instrument_list = il.get_instrument_list()

# tickers = pd.read_csv('/Users/suganeshr/Trading/Trading_codes/FnO.csv')['SYMBOL'].to_list()
tickers = ['RELIANCE','SBIN','ABB','HDFCBANK']
pos_size = 50000
hi_lo_prices = {}

        
        
def hist_data_0920(tickers,duration,interval,instrument_list,exchange="NSE"):
    hist_data_tickers = {}
    
    for ticker in tickers:
        print(f'The ticker is: {ticker}')
        token =  il.token_lookup(ticker=ticker)
        print(f'The token is: {token}')
        time.sleep(0.4)
        params = {
                 "exchange": exchange,
                 "symboltoken":token,
                 "interval": interval,
                 "fromdate": (dt.date.today() - dt.timedelta(duration)).strftime('%Y-%m-%d %H:%M'),
                 "todate": dt.date.today().strftime('%Y-%m-%d') + ' 09:19' #there seems to be a bug in smartAPI's historical data call as it also providing candles starting from to_data
                 }
        hist_data = obj.getCandleData(params)
        df_data = pd.DataFrame(hist_data["data"],
                               columns = ["date","open","high","low","close","volume"])
        df_data.set_index("date",inplace=True)
        df_data.index = pd.to_datetime(df_data.index)
        df_data.index = df_data.index.tz_localize(None)
        df_data["gap"] = ((df_data["open"]/df_data["close"].shift(1))-1)*100
        hist_data_tickers[ticker] = df_data
    return hist_data_tickers




def preprocess_data(data_dict):
    # Dictionary to store processed data for each ticker
    processed_data = {}

    # Iterate over each ticker's DataFrame in the dictionary
    for ticker, data in data_dict.items():
        # Step 1: Resample to hourly data and calculate RSI
        data_hourly = custom_resample(data, freq='1h')
        data_hourly['RSI'] = ta.rsi(data_hourly['close'], timeperiod=14)

        # Step 2: Add Date and datetime columns to the original data
        data['Date'] = data.index.date
        data['datetime'] = data.index

        # Step 3: Merge hourly RSI data into the original data
        merged_data = pd.merge(data, data_hourly[['RSI']], left_index=True, right_index=True, how='left').ffill()

        # Step 4: Resample to daily data and calculate pivot points
        daily_data = custom_resample(data, freq='D')
        daily_data['Date'] = daily_data.index.date
        daily_data = apply_pivots(daily_data)

        # Step 5: Merge daily pivot point data into the merged data
        merged_data = pd.merge(merged_data, daily_data[['Date', 'PP', 'TC', 'BC', 'R1', 'R2', 'R3', 'S1', 'S2', 'S3', '%CPR']],
                               left_on='Date', right_on='Date', how='left')

        # Step 6: Add Volume-Weighted Moving Average (VWMA)
        merged_data['wma'] = ta.vwma(merged_data['close'], merged_data['volume'], length=5)

        # Step 7: Set the index to the datetime column
        merged_data.set_index('datetime', inplace=True)
        print(f'The columns are {merged_data.columns}')
        # Store the processed data for this ticker
        processed_data[ticker] = merged_data

    return processed_data  # Return the dictionary of processed DataFrames

# Define the trading conditions based on your strategy
def define_trading_conditions(data):
    """Define the buy and sell conditions."""
    start_time = dt.time(10, 14)
    end_time = dt.time(14, 16)
    buy_condition = (
        (data['%CPR'] < 0.01) &
        (data.RSI > 70) &
        (data.close > data.R1) &
        (data.close.shift(1) < data.R1) &
        (data.index.time >= start_time) &
        (data.index.time <= end_time)
    )
                    
    sell_condition = (data.close > data.R3) | (data.close < data.TC) | (data.RSI < 50)
    short_condition = (
        (data['%CPR'] < 0.01) &
        (data.RSI < 30) &
        (data.close < data.S1) &
        (data.close.shift(1) > data.S1) &
        (data.index.time >= start_time) &
        (data.index.time <= end_time)
    )
    
    cover_condition  = (data.close < data.S3) | (data.close > data.BC) | (data.RSI > 50)
    
    data['Signal'] = np.select([buy_condition, sell_condition, short_condition, cover_condition], ['Buy', 'Sell', 'Short', 'Cover'])
    data['Shifted_close'] = data['close'].shift()

    return data
# Function to append signals to a CSV file
def append_to_csv(signal_data):
    file_exists = os.path.isfile('trading_signals.csv')
    with open('trading_signals.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['Datetime', 'Ticker', 'Signal', 'Close Price'])
        writer.writerow(signal_data)

# Main workflow logic (place orders and execute strategy based on signals)
def execute_strategy():
    data_0920 = hist_data_0920(tickers, 40, "FIVE_MINUTE", instrument_list)
    data_processed = preprocess_data(data_0920)  # Returns a dictionary of DataFrames

    # Dictionary to track positions for each ticker (None, 'long', or 'short')
    position_tracker = {ticker: None for ticker in tickers}

    # Iterate over each ticker's processed data
    for ticker, df in data_processed.items():
        print(f"Processing ticker: {ticker}")
        
        # Pass each ticker's DataFrame to define_trading_conditions
        data_with_conditions = define_trading_conditions(df)

        for index, row in data_with_conditions.iterrows():
            signal_data = [index, ticker, row['Signal'], row['close']]  # Include ticker in signal data
            
            # Handle Buy signals (can only execute if no open position)
            if row['Signal'] == 'Buy' and position_tracker[ticker] is None:
                append_to_csv(signal_data)
                print(f"Buy Signal for {ticker} at {index} for {row['close']}")
                position_tracker[ticker] = 'long'  # Update position to 'long'

            # Handle Sell signals (only if a long position is open)
            elif row['Signal'] == 'Sell' and position_tracker[ticker] == 'long':
                append_to_csv(signal_data)
                print(f"Sell Signal for {ticker} at {index} for {row['close']}")
                position_tracker[ticker] = None  # close the long position (reset to None)

            # Handle Short signals (can only execute if no open position)
            elif row['Signal'] == 'Short' and position_tracker[ticker] is None:
                append_to_csv(signal_data)
                print(f"Short Signal for {ticker} at {index} for {row['close']}")
                position_tracker[ticker] = 'short'  # Update position to 'short'

            # Handle Cover signals (only if a short position is open)
            elif row['Signal'] == 'Cover' and position_tracker[ticker] == 'short':
                append_to_csv(signal_data)
                print(f"Cover Signal for {ticker} at {index} for {row['close']}")
                position_tracker[ticker] = None  # close the short position (reset to None)

execute_strategy()
