import pandas as pd
import itertools

class ManualBacktester:
    def __init__(self, data, initial_capital=100000, commission=0.002):
        self.data = data
        self.initial_capital = initial_capital
        self.commission = commission
        self.positions = []
        self.trades = []
        self.capital = initial_capital
        self.equity_curve = []

    def calculate_indicators(self, ma_length=45, atr_length=50):
        """Calculate necessary technical indicators using basic Pandas."""
        self.data['ma'] = self.data['Close'].rolling(window=ma_length).mean()
        self.data['tr'] = self.data['High'].combine(self.data['Low'], max) - self.data['Low'].combine(self.data['Close'].shift(), min)
        self.data['atr'] = self.data['tr'].rolling(window=atr_length).mean()

    def generate_signals(self):
        """Generate buy/sell signals based on the Double Inside Bar strategy."""
        double_inside_bar = (
            (self.data['High'].shift(2) > self.data['High'].shift(1)) &
            (self.data['High'].shift(2) > self.data['High']) &
            (self.data['Low'].shift(2) < self.data['Low'].shift(1)) &
            (self.data['Low'].shift(2) < self.data['Low'])
        )

        self.data['signal'] = 0
        self.data.loc[(double_inside_bar) & (self.data['Close'] > self.data['ma']), 'signal'] = 1  # Buy Signal
        self.data.loc[(double_inside_bar) & (self.data['Close'] < self.data['ma']), 'signal'] = -1  # Sell Signal
    
    def backtest(self, atr_factor=2.5, take_profit_ratio=2.0):
        """Execute the backtest based on generated signals."""
        self.positions = []
        self.trades = []
        self.capital = self.initial_capital
        self.equity_curve = []

        for i in range(1, len(self.data)):
            if self.data['signal'].iloc[i] == 1:  # Buy Signal
                buy_price = self.data['High'].iloc[i]
                stop_loss = self.data['atr'].iloc[i] * atr_factor
                take_profit = stop_loss * take_profit_ratio

                trade = {
                    'type': 'buy',
                    'entry_price': buy_price,
                    'stop_loss': buy_price - stop_loss,
                    'take_profit': buy_price + take_profit,
                    'size': self.capital / buy_price,
                    'entry_time': self.data.index[i]
                }
                self.positions.append(trade)

            elif self.data['signal'].iloc[i] == -1:  # Sell Signal
                sell_price = self.data['Low'].iloc[i]
                stop_loss = self.data['atr'].iloc[i] * atr_factor
                take_profit = stop_loss * take_profit_ratio

                trade = {
                    'type': 'sell',
                    'entry_price': sell_price,
                    'stop_loss': sell_price + stop_loss,
                    'take_profit': sell_price - take_profit,
                    'size': self.capital / sell_price,
                    'entry_time': self.data.index[i]
                }
                self.positions.append(trade)

            # Process open trades
            self.process_trades(i)

        return self.capital - self.initial_capital  # Return net profit

    def process_trades(self, index):
        """Evaluate open trades for stop loss or take profit."""
        for position in self.positions:
            if position['type'] == 'buy':
                if self.data['Low'].iloc[index] <= position['stop_loss']:
                    # Stop Loss Hit
                    self.close_position(position, position['stop_loss'], 'stop_loss', index)
                elif self.data['High'].iloc[index] >= position['take_profit']:
                    # Take Profit Hit
                    self.close_position(position, position['take_profit'], 'take_profit', index)

            elif position['type'] == 'sell':
                if self.data['High'].iloc[index] >= position['stop_loss']:
                    # Stop Loss Hit
                    self.close_position(position, position['stop_loss'], 'stop_loss', index)
                elif self.data['Low'].iloc[index] <= position['take_profit']:
                    # Take Profit Hit
                    self.close_position(position, position['take_profit'], 'take_profit', index)

        # Update equity curve
        self.update_equity_curve()

    def close_position(self, position, exit_price, reason, index):
        """Close an open position."""
        trade_result = {
            'entry_price': position['entry_price'],
            'exit_price': exit_price,
            'size': position['size'],
            'profit': (exit_price - position['entry_price']) * position['size'] if position['type'] == 'buy' else (position['entry_price'] - exit_price) * position['size'],
            'reason': reason,
            'entry_time': position['entry_time'],
            'exit_time': self.data.index[index]
        }
        self.trades.append(trade_result)
        self.capital += trade_result['profit'] - (exit_price * position['size'] * self.commission)
        self.positions.remove(position)

    def update_equity_curve(self):
        """Update the equity curve based on the current capital."""
        self.equity_curve.append(self.capital)

    def optimize(self, ma_lengths, atr_lengths, atr_factors, take_profit_ratios):
        """Optimize the strategy over a range of parameters."""
        best_params = None
        best_performance = float('-inf')
        results = []

        for ma_length, atr_length, atr_factor, take_profit_ratio in itertools.product(ma_lengths, atr_lengths, atr_factors, take_profit_ratios):
            # Calculate indicators with the current parameters
            self.calculate_indicators(ma_length=ma_length, atr_length=atr_length)
            # Generate signals
            self.generate_signals()
            # Run backtest with the current parameters
            performance = self.backtest(atr_factor=atr_factor, take_profit_ratio=take_profit_ratio)
            # Store the results
            results.append((ma_length, atr_length, atr_factor, take_profit_ratio, performance))

            if performance > best_performance:
                best_performance = performance
                best_params = (ma_length, atr_length, atr_factor, take_profit_ratio)

        return best_params, best_performance, results

# Load your data

# Initialize the backtester
