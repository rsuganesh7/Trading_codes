# -*- coding: utf-8 -*-
"""
Angel One - Opening Range Breakout strategy backtesting with KPI

Further refactored version with smaller functions.
"""

import pandas as pd
import time
from datetime import datetime, timedelta
import pandas_ta as ta
import sys
import os

sys.path.append(os.path.abspath('/Users/suganeshr/Trading/'))

from Trading_codes.Live.login_manager import LoginManager
from Trading_codes.Live.instruments_list import InstrumentDetail

lm = LoginManager()
obj, data, usr = lm.login_and_retrieve_info()

tickers = pd.read_csv('/Users/suganeshr/Trading/Trading_codes/FnO.csv')['SYMBOL'].to_list()
bktst_start_dt = "2021-06-01 09:15"
bktst_end_dt = "2024-09-06 15:30"


def custom_resample(df, start='09:15', end='15:30', freq='1h'):
    """Resample data within a given time range."""
    df = df.between_time(start, end)
    return df.resample(freq).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
    }).dropna()


def calculate_pivot_levels(high, low, close):
    """Calculate pivot levels for given high, low, and close prices."""
    pp = (high + low + close) / 3
    tc = (pp + high) / 2
    bc = (pp + low) / 2
    r1 = 2 * pp - low
    r2 = pp + (high - low)
    r3 = high + 2 * (pp - low)
    s1 = 2 * pp - high
    s2 = pp - (high - low)
    s3 = low - 2 * (high - pp)
    return pp, tc, bc, r1, r2, r3, s1, s2, s3


def apply_pivots(df):
    """Apply pivot calculations to the dataframe."""
    df['Prev_High'] = df['high'].shift(1)
    df['Prev_Low'] = df['low'].shift(1)
    df['Prev_Close'] = df['close'].shift(1)

    pivot_columns = df.apply(
        lambda row: calculate_pivot_levels(row['Prev_High'], row['Prev_Low'], row['Prev_Close']), 
        axis=1, result_type='expand'
    )

    df[['PP', 'TC', 'BC', 'R1', 'R2', 'R3', 'S1', 'S2', 'S3']] = pivot_columns
    df['%CPR'] = abs(df['TC'] - df['BC']) / df['PP']
    df['Date'] = df.index.date
    return df


def fetch_token_for_ticker(ticker):
    """Fetch the token for a given ticker."""
    token = id.token_lookup(ticker=ticker)
    print(f'Token for {ticker}: {token}')
    return token


def create_params_for_ticker(token, interval, exchange):
    """Create request parameters for fetching historical data."""
    return {
        "exchange": exchange,
        "symboltoken": token,
        "interval": interval,
        "fromdate": bktst_start_dt,
        "todate": bktst_end_dt
    }


def fetch_historical_data_for_ticker(ticker, token, params):
    """Fetch historical data for a single ticker."""
    try:
        hist_data = obj.getCandleData(params)
        df = pd.DataFrame(hist_data["data"], columns=["date", "open", "high", "low", "close", "volume"])
        df.set_index("date", inplace=True)
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df["gap"] = ((df["open"] / df["close"].shift(1)) - 1) * 100
        df["avvol"] = df["volume"].rolling(10).mean().shift(1)
        df = apply_pivots(df)
        return df
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None


def fetch_historical_data(tickers, instrument_list, interval="ONE_DAY", exchange="NSE"):
    """Fetch historical data for multiple tickers."""
    hist_data_tickers = {}
    for ticker in tickers:
        token = fetch_token_for_ticker(ticker)
        params = create_params_for_ticker(token, interval, exchange)
        df = fetch_historical_data_for_ticker(ticker, token, params)
        if df is not None:
            hist_data_tickers[ticker] = df
    return hist_data_tickers


def fetch_intraday_data_for_ticker(ticker, date, params):
    """Fetch intraday data for a single ticker."""
    try:
        hist_data = obj.getCandleData(params)
        df = pd.DataFrame(hist_data["data"], columns=["date", "open", "high", "low", "close", "volume"])
        df.set_index("date", inplace=True)
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df["Date"] = df.index.date
        return df
    except Exception as e:
        print(f"Error fetching intraday data for {ticker} on {date}: {e}")
        return None


def create_intraday_params(ticker, date, interval, exchange):
    """Create parameters for intraday data fetch."""
    return {
        "exchange": exchange,
        "symboltoken": id.token_lookup(ticker=ticker),
        "interval": interval,
        "fromdate": (date - timedelta(days=6)).strftime("%Y-%m-%d") + ' 09:15',
        "todate": date.strftime("%Y-%m-%d") + ' 15:30'
    }


def fetch_intraday_data(ticker, date, interval, instrument_list, exchange="NSE"):
    """Fetch intraday data for a given ticker and date."""
    params = create_intraday_params(ticker, date, interval, exchange)
    return fetch_intraday_data_for_ticker(ticker, date, params)


def preprocess_data(data):
    """Preprocess the data by resampling, applying RSI, and adding pivot levels."""
    data_hourly = custom_resample(data, freq='1h')
    data_hourly['RSI'] = ta.rsi(data_hourly['close'], timeperiod=14)

    data['Date'] = data.index.date
    data['datetime'] = data.index

    merged_data = merge_hourly_data(data, data_hourly)
    daily_data = custom_resample(data, freq='D')
    daily_data['Date'] = daily_data.index.date
    daily_data = apply_pivots(daily_data)

    return finalize_preprocessed_data(merged_data, daily_data)


def merge_hourly_data(data, data_hourly):
    """Merge hourly data with the original data."""
    return pd.merge(data, data_hourly[['RSI']], left_index=True, right_index=True, how='left').ffill()


def finalize_preprocessed_data(merged_data, daily_data):
    """Finalize the preprocessed data by merging with daily data."""
    merged_data = pd.merge(merged_data, daily_data[['Date', 'PP', 'TC', 'BC', 'R1', 'R2', 'R3', 'S1', 'S2', 'S3', '%CPR']], 
                           left_on='Date', right_on='Date', how='left')
    merged_data.set_index('datetime', inplace=True)
    merged_data['vwma'] = ta.vwma(merged_data['close'], merged_data['volume'], length=5)
    return merged_data


def find_top_gap_stocks(data):
    """Identify top gap stocks with %CPR less than 1.5%."""
    top_gap_by_date = {}
    dates = data[tickers[0]].index.to_list()

    for date in dates:
        temp = find_gap_stocks_on_date(data, date)
        if not temp.empty:
            top_gap_by_date[date] = temp.index.to_list()

    return top_gap_by_date


def find_gap_stocks_on_date(data, date):
    """Identify gap stocks for a specific date."""
    temp = pd.Series()
    for ticker in data:
        try:
            if data[ticker].loc[date, '%CPR'] < 1.5:
                temp[ticker] = data[ticker].loc[date, '%CPR']
        except KeyError:
            pass
    return temp


def execute_backtest(top_gap_by_date, candle_data):
    """Execute the backtest strategy using pivot points and top gap stocks."""
    date_stats = {}

    for date in top_gap_by_date:
        date_str = date.strftime("%Y-%m-%d %H:%M")
        date_stats[date] = {}

        for ticker in top_gap_by_date[date]:
            process_ticker_for_backtest(ticker, date, date_stats, date_str, candle_data)

    return date_stats


def process_ticker_for_backtest(ticker, date, date_stats, date_str, candle_data):
        """Process a single ticker for backtesting."""
    # try:
        intraday_df = fetch_intraday_data(ticker, date, 'FIVE_MINUTE', instrument_list)
        intraday_df = preprocess_data(intraday_df)

        open_price, direction = '', ''
        date_stats[date][ticker] = 0

        evaluate_ticker_trades(intraday_df, ticker, open_price, direction, date_stats, date, date_str, candle_data)

    # except KeyError as e:
    #     print(f"KeyError: {ticker} on {date}: {e}")
    # except Exception as e:
    #     print(f"An error occurred for {ticker} on {date}: {e}")


def evaluate_ticker_trades(intraday_df, ticker, open_price, direction, date_stats, date, date_str, candle_data):
    """Evaluate trades for a single ticker during backtesting."""
    for i in range(1, len(intraday_df[1:])):
        if not open_price:
            open_price, direction = check_entry_conditions(ticker,intraday_df, i, candle_data, date_str)

        if open_price and direction == 'long':
            ticker_return = handle_exit_conditions(intraday_df, i, open_price, 'long')
            if ticker_return is not None:
                date_stats[date][ticker] = ticker_return
                break

        if open_price and direction == 'short':
            ticker_return = handle_exit_conditions(intraday_df, i, open_price, 'short')
            if ticker_return is not None:
                date_stats[date][ticker] = ticker_return
                break

def check_entry_conditions(ticker, intraday_df, i, candle_data, date_str):
    """Check entry conditions for both long and short trades."""
    open_price, direction = '', ''
    
    # Long Entry: Check for volume, VWMA above R1, and RSI > 70
    if (intraday_df.iloc[i]["volume"] > 2 * candle_data[ticker].loc[date_str, "avvol"] / 75 and
        intraday_df.iloc[i]["vwma"] > intraday_df.iloc[i]["R1"] and
        intraday_df.iloc[i-1]["vwma"] < intraday_df.iloc[i]["R1"] and
        intraday_df.iloc[i]['RSI'] > 70):  # Changed to check RSI for the current row (i)
        
        open_price = 0.8 * intraday_df.iloc[i+1]["open"] + 0.2 * intraday_df.iloc[i+1]["high"]
        direction = 'long'

    # Short Entry: Check for volume, VWMA below S1, and RSI < 30
    elif (intraday_df.iloc[i]["volume"] > 2 * candle_data[ticker].loc[date_str, "avvol"] / 75 and
          intraday_df.iloc[i]["vwma"] < intraday_df.iloc[i]["S1"] and
          intraday_df.iloc[i-1]["vwma"] > intraday_df.iloc[i]["S1"] and
          intraday_df.iloc[i]['RSI'] < 30):  # Changed to check RSI for the current row (i)
        
        open_price = 0.8 * intraday_df.iloc[i+1]["open"] + 0.2 * intraday_df.iloc[i+1]["low"]
        direction = 'short'

    return open_price, direction

def handle_exit_conditions(df, i, open_price, direction):
    """Handle the exit conditions for long and short trades."""
    if direction == 'long':
        if df.iloc[i]["close"] > df.iloc[i]["R3"]:
            return ((df.iloc[i]["R3"]) / open_price) - 1
        elif df.iloc[i]["close"] < df.iloc[i]["TC"]:
            return (df.iloc[i]["close"] / open_price) - 1
        else:
            return (df.iloc[i]["close"] / open_price) - 1

    if direction == 'short':
        if df.iloc[i]["close"] < df.iloc[i]["S3"]:
            return 1 - (df.iloc[i]["S3"] / open_price)
        elif df.iloc[i]["close"] > df.iloc[i]["BC"]:
            return 1 - (df.iloc[i]["close"] / open_price)
        else:
            return 1 - (df.iloc[i]["close"] / open_price)


# KPIs and Performance Metrics
def abs_return(date_stats):
    """Calculate the absolute return of the strategy."""
    df = pd.DataFrame(date_stats).T
    df["ret"] = 1 + df.mean(axis=1)
    return (df["ret"].cumprod() - 1).iloc[-1]


def win_rate(date_stats):
    """Calculate the win rate of the strategy."""
    win_count = sum(1 for i in date_stats for t in date_stats[i] if date_stats[i][t] > 0)
    total_count = sum(1 for i in date_stats for t in date_stats[i])
    return (win_count / total_count) * 100


def mean_return(date_stats, condition):
    """Calculate the mean return for winning or losing trades."""
    returns = [date_stats[i][t] for i in date_stats for t in date_stats[i] if condition(date_stats[i][t])]
    return sum(returns) / len(returns)


def return_curve(date_stats):
    """Plot the cumulative return curve."""
    df = pd.DataFrame(date_stats).T
    df["ret"] = 1 + df.mean(axis=1)
    df["ret"].fillna(1, inplace=True)
    df["cum_ret"] = df["ret"].cumprod() - 1
    df["cum_ret"].plot(title="Return Profile")


# Main code execution
id = InstrumentDetail()
instrument_list = id.get_instrument_list()
candle_data = fetch_historical_data(tickers, instrument_list)

top_gap_by_date = find_top_gap_stocks(candle_data)
date_stats = execute_backtest(top_gap_by_date, candle_data)

# KPIs
print("**********Strategy Performance Statistics**********")
print(f"Total cumulative return = {abs_return(date_stats):.4f}")
print(f"Total win rate = {win_rate(date_stats):.2f}%")
print(f"Mean return per win trade = {mean_return(date_stats, lambda x: x > 0):.4f}")
print(f"Mean return per loss trade = {mean_return(date_stats, lambda x: x < 0):.4f}")
return_curve(date_stats)