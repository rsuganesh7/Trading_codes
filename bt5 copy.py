#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import pandas_ta as ta
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf


# In[ ]:


data =  pd.read_csv('nifty_200/5M/ADANIPOWER.csv')


# In[ ]:


data.tail()


# In[ ]:


def calculate_macd(data, fast, slow, signal):
    data['MACD_F'] = data.ta.macd(fast=fast, slow=slow, signal=signal)[f'MACD_{fast}_{slow}_{signal}']
    data['MACD_S'] = data.ta.macd(fast=fast, slow=slow, signal=signal)[f'MACDs_{fast}_{slow}_{signal}']
    data['MACD_H'] = data.ta.macd(fast=fast, slow=slow, signal=signal)[f'MACDh_{fast}_{slow}_{signal}']
    return data


# In[ ]:


def custom_resample(df, start='09:15', end='15:30', freq='1f'):
    df = df.between_time(start, end)
    return df.resample(freq, offset='0h15min').agg(
        {
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum',
        }
    )


# In[ ]:


def calculate_pivot_levels(high, low, close):
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


# In[ ]:


def apply_pivots(df):
    # Shift OHLC columns to get yesterday's data
    df['Prev_High'] = df['High'].shift(1)
    df['Prev_Low'] = df['Low'].shift(1)
    df['Prev_Close'] = df['Close'].shift(1)

    # Apply the function to calculate pivot levels using yesterday's data
    df[['PP', 'TC', 'BC', 'R1', 'R2', 'R3', 'S1', 'S2', 'S3']] = df.apply(
        lambda row: calculate_pivot_levels(row['Prev_High'], row['Prev_Low'], row['Prev_Close']), 
        axis=1, 
        result_type='expand'
    )
    return df


def data_preprocessing(data):
    temp_Data = data.copy()
    temp_Data['Datetime'] = pd.to_datetime(temp_Data['Datetime'])
    temp_Data.set_index('Datetime', inplace=True)
    daily_data = custom_resample(temp_Data, freq='D')
    data_with_pivots = apply_pivots(daily_data)
    
    print(data.columns)
    # Add Date column to data_with_pivots
    data_with_pivots['Date'] = data_with_pivots.index.date

    data['Date'] = temp_Data.index.date
    data_with_pivots['%CPR'] = abs(data_with_pivots['TC'] - data_with_pivots['BC']) / data_with_pivots['PP']
    data_with_pivots.dropna(inplace=True)

    # Merge on the Date column
    merged_df = pd.merge(data, data_with_pivots[['Date', 'PP', 'TC', 'BC', 'R1', 'R2', 'R3', 'S1', 'S2', 'S3', '%CPR']],
                         left_on='Date', right_on='Date', how='left')
    
    merged_df['VWMA_10'] = ta.vwma(merged_df['Close'], volume=merged_df['Volume'], length=10)
    merged_df['Datetime'] = pd.to_datetime(merged_df['Datetime'])
    return merged_df.set_index('Datetime')


processed_Data = data_preprocessing(data).fillna(method='ffill').dropna()




fully_merged = processed_Data



# Define target and stop-loss percentages
target_pct = 0.02  # 2% target
stop_loss_pct = 0.01  # 1% stop loss

# Initialize variables to store entry prices
buy_entry_price = None
short_entry_price = None



# In[ ]:


# Buy Entry Condition
fully_merged['Buy_Entry'] = (
    (fully_merged['%CPR'] < 0.01)& 
    (fully_merged['Close'] > fully_merged['R1']) &
    (fully_merged['VWMA_10'].shift(1) <= fully_merged['R1'].shift(1)) & 
    (fully_merged['VWMA_10'] > fully_merged['R1']) 
)

# Short Entry Condition
fully_merged['Short_Entry'] = (
    (fully_merged['%CPR'] < 0.01)& 
    (fully_merged['Close'] < fully_merged['S1']) &
    (fully_merged['VWMA_10'].shift(1) >= fully_merged['S1'].shift(1)) & 
    (fully_merged['VWMA_10'] < fully_merged['S1']) 
)


# In[ ]:


# Calculate entry prices and set target/stop-loss based on entry prices
for i in range(len(fully_merged)):
    if fully_merged['Buy_Entry'].iloc[i]:
        buy_entry_price = fully_merged['Close'].iloc[i]
        fully_merged.loc[fully_merged.index[i], 'Buy_Target'] = buy_entry_price * (1 + target_pct)
        fully_merged.loc[fully_merged.index[i], 'Buy_Stop_Loss'] = buy_entry_price * (1 - stop_loss_pct)
    elif fully_merged['Short_Entry'].iloc[i]:
        short_entry_price = fully_merged['Close'].iloc[i]
        fully_merged.loc[fully_merged.index[i], 'Short_Target'] = short_entry_price * (1 - target_pct)
        fully_merged.loc[fully_merged.index[i], 'Short_Stop_Loss'] = short_entry_price * (1 + stop_loss_pct)
    else:
        # Carry forward the last calculated target and stop-loss
        if buy_entry_price is not None:
            fully_merged.loc[fully_merged.index[i], 'Buy_Target'] = fully_merged['Buy_Target'].iloc[i-1]
            fully_merged.loc[fully_merged.index[i], 'Buy_Stop_Loss'] = fully_merged['Buy_Stop_Loss'].iloc[i-1]
        if short_entry_price is not None:
            fully_merged.loc[fully_merged.index[i], 'Short_Target'] = fully_merged['Short_Target'].iloc[i-1]
            fully_merged.loc[fully_merged.index[i], 'Short_Stop_Loss'] = fully_merged['Short_Stop_Loss'].iloc[i-1]


# In[ ]:


# Buy Exit Condition
fully_merged['Buy_Exit'] = (
    (fully_merged['Close'] < fully_merged['TC']) |
    (fully_merged['VWMA_10'].shift(1) >= fully_merged['PP'].shift(1)) & 
    (fully_merged['VWMA_10'] < fully_merged['PP'])|
    (fully_merged['Close'] >= fully_merged['Buy_Target']) |  # Target Condition
    (fully_merged['Close'] <= fully_merged['Buy_Stop_Loss'])  # Stop-Loss Condition
)

# Short Exit Condition
fully_merged['Short_Exit'] = (
    (fully_merged['Close'] > fully_merged['BC']) |
    (fully_merged['VWMA_10'].shift(1) <= fully_merged['PP'].shift(1)) & 
    (fully_merged['VWMA_10'] > fully_merged['PP'])|
    (fully_merged['Close'] <= fully_merged['Short_Target']) |  # Target Condition
    (fully_merged['Close'] >= fully_merged['Short_Stop_Loss'])  # Stop-Loss Condition
)


# In[ ]:


fully_merged.tail(50)


# In[ ]:


# Initialize variables
initial_cash = 100000  # Starting with 100,000 units of currency
cash = initial_cash
position = 0  # Current position: 0 means no position, 1 means long, -1 means short
entry_price = 0
trade_history = []
risk_per_trade = 0.01  # Risk 1% of current cash balance per trade
target_pct = 0.02  # 2% target
stop_loss_pct = 0.01  # 1% stop loss
charges = 0


# In[ ]:


import pandas as pd
import numpy as np

# Assume fully_merged is your DataFrame with all the calculated conditions

# Define target and stop-loss percentages
target_pct = 0.02  # 2% target
stop_loss_pct = 0.01  # 1% stop loss
trade_charge = 40  # 40 units charge per trade

# Initialize variables
initial_cash = 100000  # Starting with 100,000 units of currency
cash = initial_cash
position = 0  # Current position: 0 means no position, 1 means long, -1 means short
buy_entry_price = None
short_entry_price = None
trade_history = []

# Iterate through the DataFrame
for i in range(len(fully_merged)):
    # Set up target and stop-loss for Buy Entry
    if fully_merged['Buy_Entry'].iloc[i] and position == 0:
        buy_entry_price = fully_merged['Close'].iloc[i]
        target_price = buy_entry_price * (1 + target_pct)
        stop_loss_price = buy_entry_price * (1 - stop_loss_pct)
        position = 1
        position_size = cash * 0.01 / (buy_entry_price - stop_loss_price)
        cash -= buy_entry_price * position_size + trade_charge  # Include trade charge
        charges += trade_charge
        trade_history.append({
            'Datetime': fully_merged.index[i], 
            'Type': 'Buy', 
            'Price': buy_entry_price, 
            'Size': position_size,
            'Target': target_price,
            'Stop_Loss': stop_loss_price
        })

    # Set up target and stop-loss for Short Entry
    elif fully_merged['Short_Entry'].iloc[i] and position == 0:
        short_entry_price = fully_merged['Close'].iloc[i]
        target_price = short_entry_price * (1 - target_pct)
        stop_loss_price = short_entry_price * (1 + stop_loss_pct)
        position = -1
        position_size = cash * 0.01 / (stop_loss_price - short_entry_price)
        cash += short_entry_price * position_size - trade_charge  # Include trade charge
        charges += trade_charge
        trade_history.append({
            'Datetime': fully_merged.index[i], 
            'Type': 'Short', 
            'Price': short_entry_price, 
            'Size': position_size,
            'Target': target_price,
            'Stop_Loss': stop_loss_price
        })

    # Exit conditions for Buy position
    if position == 1:
        if fully_merged['Close'].iloc[i] >= target_price:  # Target hit
            exit_price = target_price
            cash += exit_price * position_size - trade_charge  # Include trade charge
            position = 0
            profit = (exit_price - buy_entry_price) * position_size - trade_charge
            trade_history.append({
                'Datetime': fully_merged.index[i], 
                'Type': 'Sell (Target)', 
                'Price': exit_price, 
                'Profit': profit
            })
        elif fully_merged['Close'].iloc[i] <= stop_loss_price:  # Stop-loss hit
            exit_price = stop_loss_price
            cash += exit_price * position_size - trade_charge  # Include trade charge
            position = 0
            profit = (exit_price - buy_entry_price) * position_size - trade_charge
            trade_history.append({
                'Datetime': fully_merged.index[i], 
                'Type': 'Sell (Stop Loss)', 
                'Price': exit_price, 
                'Profit': profit
            })
        elif fully_merged['Buy_Exit'].iloc[i]:  # Regular exit condition
            exit_price = fully_merged['Close'].iloc[i]
            cash += exit_price * position_size - trade_charge  # Include trade charge
            position = 0
            profit = (exit_price - buy_entry_price) * position_size - trade_charge
            trade_history.append({
                'Datetime': fully_merged.index[i], 
                'Type': 'Sell (Exit)', 
                'Price': exit_price, 
                'Profit': profit
            })

    # Exit conditions for Short position
    if position == -1:
        if fully_merged['Close'].iloc[i] <= target_price:  # Target hit
            exit_price = target_price
            cash -= exit_price * position_size + trade_charge  # Include trade charge
            position = 0
            profit = (short_entry_price - exit_price) * position_size - trade_charge
            trade_history.append({
                'Datetime': fully_merged.index[i], 
                'Type': 'Cover (Target)', 
                'Price': exit_price, 
                'Profit': profit
            })
        elif fully_merged['Close'].iloc[i] >= stop_loss_price:  # Stop-loss hit
            exit_price = stop_loss_price
            cash -= exit_price * position_size + trade_charge  # Include trade charge
            position = 0
            profit = (short_entry_price - exit_price) * position_size - trade_charge
            trade_history.append({
                'Datetime': fully_merged.index[i], 
                'Type': 'Cover (Stop Loss)', 
                'Price': exit_price, 
                'Profit': profit
            })
        elif fully_merged['Short_Exit'].iloc[i]:  # Regular exit condition
            exit_price = fully_merged['Close'].iloc[i]
            cash -= exit_price * position_size + trade_charge  # Include trade charge
            position = 0
            profit = (short_entry_price - exit_price) * position_size - trade_charge
            trade_history.append({
                'Datetime': fully_merged.index[i], 
                'Type': 'Cover (Exit)', 
                'Price': exit_price, 
                'Profit': profit
            })

    # Check for end of day (intraday close)
    if fully_merged.index[i].time() == pd.to_datetime("15:15").time() and position != 0:
        exit_price = fully_merged['Close'].iloc[i]
        if position == 1:  # If in a long position
            cash += exit_price * position_size - trade_charge  # Include trade charge
            trade_history.append({
                'Datetime': fully_merged.index[i], 
                'Type': 'Sell (End of Day)', 
                'Price': exit_price, 
                'Profit': (exit_price - buy_entry_price) * position_size - trade_charge
            })
        elif position == -1:  # If in a short position
            cash -= exit_price * position_size + trade_charge  # Include trade charge
            trade_history.append({
                'Datetime': fully_merged.index[i], 
                'Type': 'Cover (End of Day)', 
                'Price': exit_price, 
                'Profit': (short_entry_price - exit_price) * position_size - trade_charge
            })
        position = 0  # Reset position to zero

# Final Portfolio Value
final_value = cash

# Convert trade history to DataFrame
trade_history_df = pd.DataFrame(trade_history)

# Calculate KPIs
total_trades = len(trade_history_df) // 2  # Assuming Buy/Sell pairs or Short/Cover pairs
winning_trades = trade_history_df[trade_history_df['Profit'] > 0]
losing_trades = trade_history_df[trade_history_df['Profit'] <= 0]

total_profit = trade_history_df['Profit'].sum()
win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
average_profit = trade_history_df['Profit'].mean()
profit_factor = winning_trades['Profit'].sum() / abs(losing_trades['Profit'].sum()) if len(losing_trades) > 0 else np.inf
max_drawdown = (trade_history_df['Profit'].cumsum().cummax() - trade_history_df['Profit'].cumsum()).max()
sharpe_ratio = trade_history_df['Profit'].mean() / trade_history_df['Profit'].std() * np.sqrt(252) if trade_history_df['Profit'].std() > 0 else np.nan
sortino_ratio = trade_history_df['Profit'].mean() / trade_history_df[trade_history_df['Profit'] < 0]['Profit'].std() * np.sqrt(252) if trade_history_df[trade_history_df['Profit'] < 0]['Profit'].std() > 0 else np.nan

# Output Results
print(f"Initial Cash: {initial_cash}")
print(f"Final Portfolio Value: {final_value}")
print(f"Total Profit: {total_profit}")
print(f"Return Percentage: {((final_value - initial_cash) / initial_cash) * 100:.2f}%")
print(f"Number of Trades: {total_trades}")
print(f"Win Rate: {win_rate:.2f}")
print(f"Average Profit/Loss per Trade: {average_profit:.2f}")
print(f"Profit Factor: {profit_factor:.2f}")
print(f"Max Drawdown: {max_drawdown:.2f}")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
print(f"Sortino Ratio: {sortino_ratio:.2f}")



# In[ ]:


trade_history_df


# In[ ]:




