import pandas as pd
import numpy as np
import math
import os
import datetime
from deap import base, creator, tools, algorithms
import random
from logzero import logger, loglevel
import logzero
from st2 import custom_resample, apply_pivots
import pandas_ta as ta
import matplotlib.pyplot as plt

# Load and preprocess functions
def load_data(file_path):
    data = pd.read_csv(file_path)
    data.Datetime = pd.to_datetime(data.Datetime)
    data.set_index('Datetime', inplace=True)
    return data

def preprocess_data(data, rsi_period, vwma_length):
    data_hourly = custom_resample(data, freq='1h')
    data_hourly['RSI'] = ta.rsi(data_hourly['Close'], length=rsi_period)

    data['Date'] = data.index.date
    data['datetime'] = data.index

    merged_data = pd.merge(data, data_hourly[['RSI']], left_index=True, right_index=True, how='left').ffill()
    daily_data = custom_resample(data, freq='D')
    daily_data['Date'] = daily_data.index.date
    daily_data = apply_pivots(daily_data)

    merged_data = pd.merge(merged_data, daily_data[['Date', 'PP', 'TC', 'BC', 'R1', 'R2', 'R3', 'S1', 'S2', 'S3', '%CPR']], 
                           left_on='Date', right_on='Date', how='left')
    merged_data.set_index('datetime', inplace=True)
    merged_data['wma'] = ta.vwma(merged_data['Close'], merged_data['Volume'], length=vwma_length)
    
    return merged_data

def define_trading_conditions(data):
    """Define the buy and sell conditions."""
    start_time = datetime.time(10, 14)
    end_time = datetime.time(14, 16)
    buy_condition = (
        (data['%CPR'] < 0.01) &
        (data.RSI > 70) &
        (data.Close > data.R1) &
        (data.Close.shift(1) < data.R1) &
        (data.index.time >= start_time) &
        (data.index.time <= end_time)
    )
                    
    sell_condition = (data.Close > data.R3) | (data.Close < data.TC) 
    
    data['Signal'] = np.select([buy_condition, sell_condition], ['Buy', 'Sell'])
    data['Shifted_close'] = data['Close'].shift()
    
    return data

def backtest(df, initial_capital=100000, risk_per_trade=0.01, stop_loss=0.01, take_profit=0.03):
    """Backtest the trading strategy with given conditions."""
    position = False
    trades = {'Buy Date': [], 'Buy Price': [], 'Sell Date': [], 'Sell Price': [], 'Quantity': []}
    capital = initial_capital
    close_time = datetime.time(15, 10)
    
    for index, row in df.iterrows():
        if row['Signal'] == 'Buy' and not position:
            position = True
            risk_amount = capital * risk_per_trade
            qty = math.floor(risk_amount / (stop_loss * row['Close']))
            capital -= qty * row['Close']
            trades['Buy Date'].append(index)
            trades['Buy Price'].append(row['Close'])
            trades['Quantity'].append(qty)
            logger.info(f"Buy Signal: {index}, Price: {row['Close']}, Quantity: {qty}")

        if position:
            if row['Signal'] == 'Sell' or row['Shifted_close'] < (1 - stop_loss) * trades['Buy Price'][-1] or \
               row['Shifted_close'] > (1 + take_profit) * trades['Buy Price'][-1] or index.time() > close_time:
                trades['Sell Date'].append(index)
                trades['Sell Price'].append(row['Close'])
                capital += qty * row['Close']
                logger.info(f"Sell Signal: {index}, Price: {row['Close']}, PnL: {qty * (row['Close'] - trades['Buy Price'][-1])}")
                position = False

    # Ensure lists are of equal length by filling with NaN
    max_len = max(len(trades['Buy Date']), len(trades['Sell Date']))
    for key in trades.keys():
        trades[key].extend([np.nan] * (max_len - len(trades[key])))

    return pd.DataFrame(trades)

def calculate_trade_metrics(trade_history_df):
    """Calculate metrics like Return, PnL, Cumulative Profit, etc."""
    # Ensure 'Buy Date' and 'Sell Date' are in datetime format
    trade_history_df['Buy Date'] = pd.to_datetime(trade_history_df['Buy Date'])
    trade_history_df['Sell Date'] = pd.to_datetime(trade_history_df['Sell Date'])

    # Calculate the trading metrics
    trade_history_df['Return'] = (trade_history_df['Sell Price'] - trade_history_df['Buy Price']) / trade_history_df['Buy Price']
    trade_history_df['Days'] = (trade_history_df['Sell Date'] - trade_history_df['Buy Date']).dt.days
    trade_history_df['PnL'] = trade_history_df['Sell Price'] - trade_history_df['Buy Price']
    trade_history_df['Realized Profit'] = trade_history_df['PnL'] * trade_history_df['Quantity']
    trade_history_df['Cum Profit'] = trade_history_df['Realized Profit'].cumsum()
    trade_history_df['Cumulative Return'] = (1 + trade_history_df['Return']).cumprod() - 1
    trade_history_df['Drawdown'] = trade_history_df['Cum Profit'].cummax() - trade_history_df['Cum Profit']
    
    # Calculate KPIs
    total_trades = len(trade_history_df.dropna())
    winning_trades = len(trade_history_df[trade_history_df['PnL'] > 0].dropna())
    losing_trades = len(trade_history_df[trade_history_df['PnL'] < 0].dropna())
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    avg_pnl = trade_history_df['PnL'].mean()
    total_profit = trade_history_df['Realized Profit'].sum()
    
    kpi = {
        'Total Trades': total_trades,
        'Winning Trades': winning_trades,
        'Losing Trades': losing_trades,
        'Win Rate': win_rate,
        'Average PnL': avg_pnl,
        'Total Profit': total_profit,
        'Max Drawdown': trade_history_df['Drawdown'].max()
    }
    
    return trade_history_df, kpi

# Genetic Algorithm Implementation

def process_file(file_paths, rsi_period, vwma_length, stop_loss, take_profit):
    total_profits = []
    results_dir = "backtest_results"
    os.makedirs(results_dir, exist_ok=True)

    for file_path in file_paths:
        file_name = os.path.basename(file_path).replace('.csv', '')
        data = load_data(file_path)
        processed_data = preprocess_data(data, rsi_period, vwma_length)
        signal_data = define_trading_conditions(processed_data)
        backtest_results = backtest(signal_data, stop_loss=stop_loss, take_profit=take_profit)
        trade_history_df, kpi = calculate_trade_metrics(backtest_results)
        
        # Save the backtest results for this file
        result_file_name = f"{file_name}_RSI{rsi_period}_VWMA{vwma_length}_SL{stop_loss}_TP{take_profit}.csv"
        result_file_path = os.path.join(results_dir, result_file_name)
        trade_history_df.to_csv(result_file_path, index=False)
        
        total_profits.append(kpi['Total Profit'])

    # Return the average total profit across all files
    return np.mean(total_profits)

def genetic_algorithm(file_paths):
    # Define parameter ranges
    rsi_periods = range(8, 15)
    vwma_lengths = range(3, 6)
    stop_losses = [0.01, 0.02, 0.03]
    take_profits = [0.02, 0.03, 0.04]

    # Define individual and fitness
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_rsi", random.choice, rsi_periods)
    toolbox.register("attr_vwma", random.choice, vwma_lengths)
    toolbox.register("attr_sl", random.choice, stop_losses)
    toolbox.register("attr_tp", random.choice, take_profits)
    toolbox.register("individual", tools.initCycle, creator.Individual,
                     (toolbox.attr_rsi, toolbox.attr_vwma, toolbox.attr_sl, toolbox.attr_tp), n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def evaluate(individual):
        rsi_period, vwma_length, stop_loss, take_profit = individual
        avg_profit = process_file(file_paths=file_paths, 
                                  rsi_period=rsi_period, 
                                  vwma_length=vwma_length, 
                                  stop_loss=stop_loss, 
                                  take_profit=take_profit)
        return (avg_profit,)  # Return as a single-element tuple

    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    
    # Mutate with additional logging to catch complex numbers
    def safe_mutPolynomialBounded(individual, eta, low, up, indpb):
        # Log the individual before mutation
        logger.debug(f"Before mutation: {individual}")
        
        # Perform the mutation
        mutated_individual = tools.mutPolynomialBounded(individual, eta=eta, low=low, up=up, indpb=indpb)
        
        # Ensure values stay within the specified bounds and remain real numbers
        for i, value in enumerate(mutated_individual[0]):
            # Clamp the value within the bounds
            mutated_individual[0][i] = max(min(value.real, up[i]), low[i])
        
        # Log the individual after mutation
        logger.debug(f"After mutation: {mutated_individual}")
        
        return mutated_individual
    
    toolbox.register("mutate", safe_mutPolynomialBounded, eta=0.1, low=[10, 3, 0.005, 0.02], up=[20, 10, 0.015, 0.04], indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    population = toolbox.population(n=50)
    algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=20, verbose=True)

    best_individual = tools.selBest(population, k=1)[0]
    print(f"Best Parameters: {best_individual}")
    print(f"Best KPI: {evaluate(best_individual)}")

if __name__ == "__main__":
    folder_path = 'FnO/5M'
    stock_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.csv')]
    genetic_algorithm(stock_files)