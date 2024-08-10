import yfinance as yf
import pandas as pd
import pandas_ta as ta


class StockBacktester:
    def __init__(self, stock_symbol, interval="1h", period="2y"):
        self.stock_symbol = stock_symbol
        self.interval = interval
        self.period = period
        self.stock_data = None
        self.df_daily = None
        self.results = None
        self.trades = []

    def download_stock_data(self):
        """
        Download historical stock data from Yahoo Finance.
        """
        self.stock_data = yf.download(
            f"{self.stock_symbol}.NS", interval=self.interval, period=self.period
        )

    def resample_to_daily(self):
        """
        Resample the hourly stock data to daily data for pivot calculations.
        """
        self.df_daily = (
            self.stock_data.resample("D")
            .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"})
            .dropna()
        )
        self.df_daily["Date"] = self.df_daily.index.date

    def calculate_pivots(self):
        """
        Calculate daily Pivot Points, TC, BC, R1, R2, R3, S1, S2, S3
        """
        self.df_daily["PP"] = (
            self.df_daily["High"] + self.df_daily["Low"] + self.df_daily["Close"]
        ) / 3
        self.df_daily["R1"] = 2 * self.df_daily["PP"] - self.df_daily["Low"]
        self.df_daily["S1"] = 2 * self.df_daily["PP"] - self.df_daily["High"]
        self.df_daily["R2"] = (
            self.df_daily["PP"] + self.df_daily["High"] - self.df_daily["Low"]
        )
        self.df_daily["S2"] = (
            self.df_daily["PP"] - self.df_daily["High"] + self.df_daily["Low"]
        )
        self.df_daily["R3"] = (
            self.df_daily["PP"] + self.df_daily["High"] - self.df_daily["Low"] * 1.5
        )
        self.df_daily["S3"] = (
            self.df_daily["PP"] - self.df_daily["High"] + self.df_daily["Low"] * 1.5
        )
        self.df_daily["TC"] = (self.df_daily["PP"] * 2) - self.df_daily["Low"]
        self.df_daily["BC"] = (self.df_daily["PP"] * 2) - self.df_daily["High"]
        self.df_daily["%CPR"] = abs(
            (self.df_daily["TC"] - self.df_daily["BC"]) / self.df_daily["PP"] * 100
        )
        self.df_daily = self.df_daily.drop(
            columns=["Open", "High", "Low", "Close"], errors="ignore"
        )

    def add_technical_indicators(self):
        """
        Calculate VWMA and RSI indicators and add them to the stock data.
        """
        self.stock_data["VWMA"] = ta.vwma(
            self.stock_data["Close"], self.stock_data["Volume"], length=14
        )
        self.stock_data["RSI"] = ta.rsi(self.stock_data["Close"], length=14)

    def merge_pivots_with_stock_data(self):
        """
        Merge the daily pivot points with the hourly stock data.
        """
        self.stock_data = self.stock_data.merge(
            self.df_daily, left_index=True, right_index=True, how="left"
        )

    def generate_signals(self):
        """
        Generate buy and sell signals based on the trading strategy.
        """
        buy_signals = (
            # (self.stock_data['RSI'] > 60) &
            # (self.stock_data['%CPR'] < 0.5) &
            # (self.stock_data['PP'] > self.stock_data['PP'].shift(1)) &
            (self.stock_data["VWMA"] > self.stock_data["R1"])
            & (self.stock_data["VWMA"].shift(1) < self.stock_data["R1"].shift(1))
            & self.stock_data["Close"]
            > self.stock_data["VWMA"]
        )

        self.stock_data["Buy_Signal"] = buy_signals
        self.stock_data["Sell_Signal"] = False  # Placeholder for sell conditions
        print(f'The buy signal occurs on {self.stock_data["Buy_Signal"].idxmax()}')
    def execute_trades(self):
        """
        Execute trades based on the generated buy and sell signals.
        """
        position = 0
        for i in range(1, len(self.stock_data)):
            if self.stock_data["Buy_Signal"].iloc[i] and position == 0:
                # Buy the stock
                entry_price = self.stock_data["Close"].iloc[i]
                shares = self.position_size // entry_price
                position = shares * entry_price
                self.trades.append(
                    {
                        "Type": "Buy",
                        "Shares": shares,
                        "Price": entry_price,
                        "Value": position,
                        "Date": self.stock_data.index[i],
                    }
                )
            elif self.stock_data["Sell_Signal"].iloc[i] and position > 0:
                # Sell the stock
                exit_price = self.stock_data["Close"].iloc[i]
                value = shares * exit_price
                profit = value - position
                self.trades.append(
                    {
                        "Type": "Sell",
                        "Shares": shares,
                        "Price": exit_price,
                        "Value": value,
                        "Profit": profit,
                        "Date": self.stock_data.index[i],
                    }
                )
                position = 0

    def run_backtest(self):
        """
        Execute the backtesting process.
        """
        self.download_stock_data()
        self.resample_to_daily()
        self.calculate_pivots()
        self.add_technical_indicators()
        self.merge_pivots_with_stock_data()
        self.generate_signals()
        self.execute_trades()
        self.results = pd.DataFrame(self.trades)
        return self.results


class MultipleStockBacktester:
    def __init__(self, stock_list):
        self.stock_list = stock_list
        self.backtest_results = {}

    def run(self):
        """
        Run the backtest for each stock in the list.
        """
        for stock in self.stock_list:
            tester = StockBacktester(stock)
            self.backtest_results[stock] = tester.run_backtest()

    def get_results(self):
        """
        Return the results of the backtesting process.
        """
        return self.backtest_results


# Example usage
if __name__ == "__main__":
    # List of stocks to backtest
    stocks = ["SBIN", "RELIANCE", "TCS", "INFY", "HDFCBANK"]

    # Create the multiple stock backtester object
    multi_backtester = MultipleStockBacktester(stocks)

    # Run the backtests
    multi_backtester.run()

    # Get the results
    results = multi_backtester.get_results()

    # Display the results
    print(results)
