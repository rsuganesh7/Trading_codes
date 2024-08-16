import pandas as pd
import numpy as np
import pandas_ta as ta
import os

class TradingStrategy:
    def __init__(self, initial_cash, target_pct=0.02, stop_loss_pct=0.01, trade_charge=40):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.position = 0  # 1 for long, -1 for short, 0 for none
        self.trade_history = []
        self.entry_price = None
        self.target_price = None
        self.stop_loss_price = None
        self.target_pct = target_pct
        self.stop_loss_pct = stop_loss_pct
        self.trade_charge = trade_charge
    
    def execute_strategy(self, data):
        data = self.preprocess_data(data)
        for index, row in data.iterrows():
            if index > 0:  # Process only if there is a previous row
                # try:
                    prev_row = data.iloc[index - 1]
                    self.process_row(row, prev_row)
                # except Exception as e:
                #     print(f'The error {e} for index {index}' )
    def calculate_pivot_levels(self, high, low, close):
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
    
    def custom_resample(self, df, start='09:15', end='15:30', freq='1D'):
        df = df.between_time(start, end)
        return df.resample(freq).agg(
            {
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum',
            }
        ).dropna()
    
    def preprocess_data(self, df):
        data = self.custom_resample(df.copy(), freq='1D')
        data['Prev_High'] = data['High'].shift(1)
        data['Prev_Low'] = data['Low'].shift(1)
        data['Prev_Close'] = data['Close'].shift(1)
        data[['PP', 'TC', 'BC', 'R1', 'R2', 'R3', 'S1', 'S2', 'S3']] = data.apply(
            lambda row: self.calculate_pivot_levels(row['Prev_High'], row['Prev_Low'], row['Prev_Close']), 
            axis=1, 
            result_type='expand'
        )
        data['VWMA_10'] = ta.vwma(data['Close'], volume=data['Volume'], length=10)
        data['Supertrend'] = data.ta.supertrend(length=10, multiplier=4)['SUPERT_10_4.0']
        data['%CPR'] = abs(data['TC'] - data['BC'])/data['PP']
        df['Date'] = df.index.date
        df['VWMA_10'] = ta.vwma(df['Close'], volume=df['Volume'], length=10)
        data['Date'] = data.index.date
        fully_merged = pd.merge(df.dropna(), data[['Date', 'PP', 'TC', 'BC', 'R1', 'R2', 'R3', 'S1', 'S2', 'S3', '%CPR']].dropna(), on='Date', how='left')
        fully_merged['%CPR_condition'] = fully_merged['%CPR'] < 0.01
        print(fully_merged.columns)
        return fully_merged.dropna()

    def process_row(self, row, prev_row):
        if self.position == 0:
            if self.buy_signal(row, prev_row):
                self.enter_position(row, "Buy")
            elif self.short_signal(row, prev_row):
                self.enter_position(row, "Short")

        if self.position == 1:
            if self.check_exit_conditions(row, "Buy"):
                self.exit_position(row, "Sell")

        elif self.position == -1:
            if self.check_exit_conditions(row, "Short"):
                self.exit_position(row, "Cover")

        # Check for end of day (intraday close)
        if row.name.time() == pd.to_datetime("15:15").time() and self.position != 0:
            self.exit_position(row, "End of Day")

    def buy_signal(self, row, prev_row):
        return (
            (row['Close'] > row['R1']) and
            (row['VWMA_10'] > row['R1']) and
            (prev_row['VWMA_10'] <= prev_row['R1']) and
            row['%CPR_condition']
        )

    def sell_signal(self, row):
        return (
            (row['Close'] < row['TC']) or
            (row['VWMA_10'] < row['PP'])
        )

    def short_signal(self, row, prev_row):
        return (
            (row['Close'] < row['S1']) and
            (row['VWMA_10'] < row['S1']) and
            (prev_row['VWMA_10'] >= prev_row['S1']) and
            row['%CPR_condition']
        )

    def cover_signal(self, row):
        return (
            (row['Close'] > row['BC']) or
            (row['VWMA_10'] > row['PP'])
        )

    def check_exit_conditions(self, row, position_type):
        if position_type == "Buy":
            return (
                (row['Close'] >= self.target_price) or
                (row['Close'] <= self.stop_loss_price) or
                self.sell_signal(row)
            )
        elif position_type == "Short":
            return (
                (row['Close'] <= self.target_price) or
                (row['Close'] >= self.stop_loss_price) or
                self.cover_signal(row)
            )

    def enter_position(self, row, action):
        entry_price = row['Close']
        if action == "Buy":
            self.target_price = entry_price * (1 + self.target_pct)
            self.stop_loss_price = entry_price * (1 - self.stop_loss_pct)
            position_size = self.cash * 0.01 / (entry_price - self.stop_loss_price)
            self.cash -= entry_price * position_size + self.trade_charge
            self.position = 1
        elif action == "Short":
            self.target_price = entry_price * (1 - self.target_pct)
            self.stop_loss_price = entry_price * (1 + self.stop_loss_pct)
            position_size = self.cash * 0.01 / (self.stop_loss_price - entry_price)
            self.cash += entry_price * position_size - self.trade_charge
            self.position = -1

        self.entry_price = entry_price
        self.trade_history.append({
            'Datetime': row.name,
            'Action': action,
            'Price': entry_price,
            'Size': position_size,
            'Target': self.target_price,
            'Stop_Loss': self.stop_loss_price
        })

    def exit_position(self, row, action):
        exit_price = row['Close']
        last_trade = self.trade_history[-1]
        position_size = last_trade['Size']
        profit = 0

        if self.position == 1:  # Closing a Buy position
            self.cash += exit_price * position_size - self.trade_charge
            profit = (exit_price - self.entry_price) * position_size - self.trade_charge

        elif self.position == -1:  # Closing a Short position
            self.cash -= exit_price * position_size + self.trade_charge
            profit = (self.entry_price - exit_price) * position_size - self.trade_charge

        self.trade_history.append({
            'Datetime': row.name,
            'Action': action,
            'Price': exit_price,
            'Profit': profit
        })

        self.position = 0  # Reset position to zero
        self.entry_price = None
        self.target_price = None
        self.stop_loss_price = None



class TradeMetrics:
    def __init__(self, trade_history, initial_cash, final_cash):
        self.trade_history = pd.DataFrame(trade_history)
        self.initial_cash = initial_cash
        self.final_cash = final_cash
        
    def calculate_metrics(self):
        total_trades = len(self.trade_history) // 2  # Assuming Buy/Sell pairs or Short/Cover pairs
        winning_trades = self.trade_history[self.trade_history['Profit'] > 0]
        losing_trades = self.trade_history[self.trade_history['Profit'] <= 0]
        
        total_profit = self.trade_history['Profit'].sum()
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        average_profit = self.trade_history['Profit'].mean()
        profit_factor = winning_trades['Profit'].sum() / abs(losing_trades['Profit'].sum()) if len(losing_trades) > 0 else np.inf
        max_drawdown = (self.trade_history['Profit'].cumsum().cummax() - self.trade_history['Profit'].cumsum()).max()
        sharpe_ratio = self.trade_history['Profit'].mean() / self.trade_history['Profit'].std() * np.sqrt(252) if self.trade_history['Profit'].std() > 0 else np.nan
        sortino_ratio = self.trade_history['Profit'].mean() / self.trade_history[self.trade_history['Profit'] < 0]['Profit'].std() * np.sqrt(252) if self.trade_history[self.trade_history['Profit'] < 0]['Profit'].std() > 0 else np.nan
        
        return {
            'Initial Cash': self.initial_cash,
            'Final Portfolio Value': self.final_cash,
            'Total Profit': total_profit,
            'Return Percentage': ((self.final_cash - self.initial_cash) / self.initial_cash) * 100,
            'Number of Trades': total_trades,
            'Win Rate': win_rate,
            'Average Profit/Loss per Trade': average_profit,
            'Profit Factor': profit_factor,
            'Max Drawdown': max_drawdown,
            'Sharpe Ratio': sharpe_ratio,
            'Sortino Ratio': sortino_ratio
        }
        
def backtest_multiple_stocks(stock_files, initial_cash):
    overall_results = []
    for stock_file in stock_files:
        # try: 
            file_name = os.path.basename(stock_file)
            data = pd.read_csv(stock_file, index_col='Datetime', parse_dates=True)
            stock_name = os.path.basename(stock_file).replace('.csv', '')
            
            strategy = TradingStrategy(initial_cash)
            strategy.execute_strategy(data)
            metrics = TradeMetrics(strategy.trade_history, initial_cash, strategy.cash)
            trade_history_df = pd.DataFrame(strategy.trade_history)
            trade_history_df.to_csv(os.path.join('Backtest/Results',f'{file_name}'))

            results = metrics.calculate_metrics()
            results['Stock'] = stock_name
            overall_results.append(results)
        # except Exception as e:
        #     print(f'Error {e} on {file_name}')

    return pd.DataFrame(overall_results)

def main():
    directory = 'nifty_200/5M'
    initial_cash = 100000  # Starting with $100,000 for each stock
    stock_files = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith('.csv')]
    # try:
    results_df = backtest_multiple_stocks(stock_files, initial_cash)
    results_df.to_csv('results.csv')
    print(results_df)
# except Exception as e:
    # print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
