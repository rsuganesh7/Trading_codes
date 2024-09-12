import os
import pandas as pd
import numpy as np
import pandas_ta as ta
import math
import plotly.graph_objects as go
from concurrent.futures import ThreadPoolExecutor, as_completed

def load_data(file_path):
    return pd.read_csv(file_path)

def calculate_macd(data, fast, slow, signal):
    data['MACD_F'] = data.ta.macd(fast=fast, slow=slow, signal=signal)[f'MACD_{fast}_{slow}_{signal}']
    data['MACD_S'] = data.ta.macd(fast=fast, slow=slow, signal=signal)[f'MACDs_{fast}_{slow}_{signal}']
    data['MACD_H'] = data.ta.macd(fast=fast, slow=slow, signal=signal)[f'MACDh_{fast}_{slow}_{signal}']
    return data

def custom_resample(df, start='09:15', end='15:30', freq='1h'):
    
    df = df.between_time(start, end)
    return df.resample(freq, offset='0h15min').agg(
        {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
        }
    ).dropna()

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

def apply_pivots(df):
    df['Prev_High'] = df['high'].shift(1)
    df['Prev_Low'] = df['low'].shift(1)
    df['Prev_Close'] = df['close'].shift(1)

    df[['PP', 'TC', 'BC', 'R1', 'R2', 'R3', 'S1', 'S2', 'S3']] = df.apply(
        lambda row: calculate_pivot_levels(row['Prev_High'], row['Prev_Low'], row['Prev_Close']), 
        axis=1, 
        result_type='expand'
    )
    df['%CPR'] = abs(df['TC'] - df['BC'])/df['PP']
    df['Date'] = df.index.date
    return df

def data_preprocessing(data,file_name):
    print(f'Processing data for {file_name}...')
    temp_Data = data.copy()
    temp_Data['Datetime'] = pd.to_datetime(temp_Data['Datetime'])
    temp_Data.set_index('Datetime', inplace=True)
    daily_data = custom_resample(temp_Data, freq='D')
    data_with_pivots = apply_pivots(daily_data)
    
    data_with_pivots['Date'] = data_with_pivots.index.date
    data['Date'] = temp_Data.index.date
    data_with_pivots['%CPR'] = abs(data_with_pivots['TC'] - data_with_pivots['BC']) / data_with_pivots['PP']
    data_with_pivots.dropna(inplace=True)

    merged_df = pd.merge(data, data_with_pivots[['Date', 'PP', 'TC', 'BC', 'R1', 'R2', 'R3', 'S1', 'S2', 'S3', '%CPR']],
                         left_on='Date', right_on='Date', how='left')
    
    merged_df['VWMA_10'] = ta.vwma(merged_df['Close'], volume=merged_df['Volume'], length=10)
    merged_df['Datetime'] = pd.to_datetime(merged_df['Datetime'])
    return merged_df.set_index('Datetime')

def generate_signals(data,file_name, target_pct=0.02, stop_loss_pct=0.01):
    print(f'Generating signals for {file_name}...')
    # Buy Entry Condition
    data['Buy_Entry'] = (
        (data['%CPR'] < 0.01) & 
        (data['Close'] > data['R1']) &
        (data['VWMA_10'].shift(1) <= data['R1'].shift(1)) & 
        (data['VWMA_10'] > data['R1'])
    )

    # Short Entry Condition
    data['Short_Entry'] = (
        (data['%CPR'] < 0.01) & 
        (data['Close'] < data['S1']) &
        (data['VWMA_10'].shift(1) >= data['S1'].shift(1)) & 
        (data['VWMA_10'] < data['S1'])
    )

    # Initialize variables to store entry prices
    buy_entry_price = None
    short_entry_price = None

    # Calculate entry prices and set target/stop-loss based on entry prices
    for i in range(len(data)):
        if data['Buy_Entry'].iloc[i]:
            buy_entry_price = data['Close'].iloc[i]
            data.loc[data.index[i], 'Buy_Target'] = buy_entry_price * (1 + target_pct)
            data.loc[data.index[i], 'Buy_Stop_Loss'] = buy_entry_price * (1 - stop_loss_pct)
        elif data['Short_Entry'].iloc[i]:
            short_entry_price = data['Close'].iloc[i]
            data.loc[data.index[i], 'Short_Target'] = short_entry_price * (1 - target_pct)
            data.loc[data.index[i], 'Short_Stop_Loss'] = short_entry_price * (1 + stop_loss_pct)
        else:
            # Carry forward the last calculated target and stop-loss
            if buy_entry_price is not None:
                data.loc[data.index[i], 'Buy_Target'] = data['Buy_Target'].iloc[i-1]
                data.loc[data.index[i], 'Buy_Stop_Loss'] = data['Buy_Stop_Loss'].iloc[i-1]
            if short_entry_price is not None:
                data.loc[data.index[i], 'Short_Target'] = data['Short_Target'].iloc[i-1]
                data.loc[data.index[i], 'Short_Stop_Loss'] = data['Short_Stop_Loss'].iloc[i-1]

    # Buy Exit Condition
    data['Buy_Exit'] = (
        (data['Close'] < data['TC']) |
        (data['VWMA_10'].shift(1) >= data['PP'].shift(1)) & 
        (data['VWMA_10'] < data['PP']) |
        (data['Close'] >= data['Buy_Target']) |  # Target Condition
        (data['Close'] <= data['Buy_Stop_Loss'])  # Stop-Loss Condition
    )

    # Short Exit Condition
    data['Short_Exit'] = (
        (data['Close'] > data['BC']) |
        (data['VWMA_10'].shift(1) <= data['PP'].shift(1)) & 
        (data['VWMA_10'] > data['PP']) |
        (data['Close'] <= data['Short_Target']) |  # Target Condition
        (data['Close'] >= data['Short_Stop_Loss'])  # Stop-Loss Condition
    )

    return data
def simulate_trading(data,file_name, initial_cash=100000, trade_charge=40):
    print(f'Simulating trading for {file_name}...')
    cash = initial_cash
    position = 0
    trade_history = []
    charges = 0
    position_size = 0

    for i in range(len(data)):
        if data['Buy_Entry'].iloc[i] and position == 0:
            position = 1
            buy_entry_price = data['Close'].iloc[i]
            position_size = math.floor(cash/buy_entry_price)
            cash -= buy_entry_price * position_size + trade_charge
            charges += trade_charge
            trade_history.append({'Datetime': data.index[i], 'Type': 'Buy', 'Price': buy_entry_price, 'Size': position_size})

        elif data['Short_Entry'].iloc[i] and position == 0:
            position = -1
            short_entry_price = data['Close'].iloc[i]
            position_size = math.floor(cash/short_entry_price)
            cash += short_entry_price * position_size - trade_charge
            charges += trade_charge
            trade_history.append({'Datetime': data.index[i], 'Type': 'Short', 'Price': short_entry_price, 'Size': position_size})

        if position == 1:
            if data['Buy_Exit'].iloc[i]:
                exit_price = data['Close'].iloc[i]
                cash += exit_price * position_size - trade_charge
                profit = (exit_price - buy_entry_price) * position_size - trade_charge
                trade_history.append({'Datetime': data.index[i], 'Type': 'Sell', 'Price': exit_price, 'Profit': profit})
                position = 0

        if position == -1:
            if data['Short_Exit'].iloc[i]:
                exit_price = data['Close'].iloc[i]
                cash -= exit_price * position_size + trade_charge
                profit = (short_entry_price - exit_price) * position_size - trade_charge
                trade_history.append({'Datetime': data.index[i], 'Type': 'Cover', 'Price': exit_price, 'Profit': profit})
                position = 0

        if data.index[i].time() == pd.to_datetime("15:15").time() and position != 0:
            exit_price = data['Close'].iloc[i]
            if position == 1:
                cash += exit_price * position_size - trade_charge
                profit = (exit_price - buy_entry_price) * position_size - trade_charge
                trade_history.append({'Datetime': data.index[i], 'Type': 'Sell (End of Day)', 'Price': exit_price, 'Profit': profit})
            elif position == -1:
                cash -= exit_price * position_size + trade_charge
                profit = (short_entry_price - exit_price) * position_size - trade_charge
                trade_history.append({'Datetime': data.index[i], 'Type': 'Cover (End of Day)', 'Price': exit_price, 'Profit': profit})
            position = 0

    final_value = cash
    trade_history_df = pd.DataFrame(trade_history)
    total_trades = len(trade_history_df) // 2
    winning_trades = trade_history_df[trade_history_df['Profit'] > 0]
    losing_trades = trade_history_df[trade_history_df['Profit'] <= 0]

    total_profit = trade_history_df['Profit'].sum()
    win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
    average_profit = trade_history_df['Profit'].mean()
    profit_factor = winning_trades['Profit'].sum() / abs(losing_trades['Profit'].sum()) if len(losing_trades) > 0 else np.inf
    max_drawdown = (trade_history_df['Profit'].cumsum().cummax() - trade_history_df['Profit'].cumsum()).max()
    sharpe_ratio = trade_history_df['Profit'].mean() / trade_history_df['Profit'].std() * np.sqrt(252) if trade_history_df['Profit'].std() > 0 else np.nan
    sortino_ratio = trade_history_df['Profit'].mean() / trade_history_df[trade_history_df['Profit'] < 0]['Profit'].std() * np.sqrt(252) if trade_history_df[trade_history_df['Profit'] < 0]['Profit'].std() > 0 else np.nan

    return {
        'Stock':file_name,  # Use filename as stock name
        'Initial Cash': initial_cash,
        'Final Portfolio Value': final_value,
        'Total Profit': total_profit,
        'Return Percentage': ((final_value - initial_cash) / initial_cash) * 100,
        'Number of Trades': total_trades,
        'Win Rate': win_rate,
        'Average Profit/Loss per Trade': average_profit,
        'Profit Factor': profit_factor,
        'Max Drawdown': max_drawdown,
        'Sharpe Ratio': sharpe_ratio,
        'Sortino Ratio': sortino_ratio
    }, trade_history_df
def plot_trades_on_chart(data, trade_history_df, file_name):
    fig = go.Figure(data=[go.Candlestick(x=data.index,
                                         open=data['Open'],
                                         high=data['High'],
                                         low=data['Low'],
                                         close=data['Close'])])

    # Add markers for trades
    for i, trade in trade_history_df.iterrows():
        if trade['Type'] == 'Buy':
            fig.add_trace(go.Scatter(x=[trade['Datetime']],
                                     y=[trade['Price']],
                                     mode='markers',
                                     marker=dict(symbol='triangle-up', color='green', size=10),
                                     name='Buy Entry'))
        elif trade['Type'] == 'Sell' or trade['Type'] == 'Sell (End of Day)':
            fig.add_trace(go.Scatter(x=[trade['Datetime']],
                                     y=[trade['Price']],
                                     mode='markers',
                                     marker=dict(symbol='triangle-down', color='red', size=10),
                                     name='Sell Exit'))
        elif trade['Type'] == 'Short':
            fig.add_trace(go.Scatter(x=[trade['Datetime']],
                                     y=[trade['Price']],
                                     mode='markers',
                                     marker=dict(symbol='triangle-down', color='orange', size=10),
                                     name='Short Entry'))
        elif trade['Type'] == 'Cover' or trade['Type'] == 'Cover (End of Day)':
            fig.add_trace(go.Scatter(x=[trade['Datetime']],
                                     y=[trade['Price']],
                                     mode='markers',
                                     marker=dict(symbol='triangle-up', color='blue', size=10),
                                     name='Cover Exit'))

    fig.update_layout(title=f'{file_name} - Trade Visualization',
                      xaxis_title='Datetime',
                      yaxis_title='Price',
                      xaxis_rangeslider_visible=False)

    # Save the plot as an HTML file
    plot_dir = 'Backtest/Plots'
    os.makedirs(plot_dir, exist_ok=True)
    fig.write_html(os.path.join(plot_dir, f'{file_name}_trade_visualization.html'))
    print(f"Trade visualization saved to '{plot_dir}/{file_name}_trade_visualization.html'")

def process_file(file_path):
    file_name = os.path.basename(file_path).replace('.csv', '')
    data = load_data(file_path)
    processed_data = data_preprocessing(data, file_name)
    processed_data = generate_signals(processed_data, file_name)
    results, trade_history_df = simulate_trading(processed_data, file_name)
    
    # Save the trade history to a CSV file
    output_dir = 'Backtest/Results/Charts'
    os.makedirs(output_dir, exist_ok=True)
    trade_history_df.to_csv(os.path.join(output_dir, f'{file_name}_trade_history.csv'), index=False)
    # Plot the trades on a candlestick chart
    plot_trades_on_chart(processed_data, trade_history_df, file_name)

    return results

def main(folder_path):
    result_list = []
    stock_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.csv')]

    with ThreadPoolExecutor() as executor:
        future_to_file = {executor.submit(process_file, file): file for file in stock_files}

        for future in as_completed(future_to_file):
            file = future_to_file[future]
            try:
                result = future.result()
                result_list.append(result)
            except Exception as e:
                print(f"An error occurred while processing {file}: {e}")

    # Save the results to a CSV file
    result_df = pd.DataFrame(result_list)
    result_df.to_csv('backtesting_results.csv', index=False)
    print("Backtesting results saved to 'backtesting_results.csv'")

if __name__ == "__main__":
    main('nifty_200/5M')

