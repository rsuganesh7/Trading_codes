"""indicators.py"""

import numpy as np


class TechnicalIndicators:
    def __init__(self, df) -> None:
        self.df = df

    def calculate_ema(self, window=12, OHLC="close"):
        return self.df[OHLC].ewm(span=window, adjust=False).mean()

    def calculate_sma(self, window=20, OHLC="close"):
        return self.df[OHLC].rolling(window=window).mean()

    def calculate_wma(self, window=20, OHLC="close"):
        weights = np.arange(1, window + 1)
        return (
            self.df[OHLC]
            .rolling(window=window)
            .apply(lambda prices: np.dot(prices, weights) / weights.sum(), raw=True)
        )

    def calculate_vwap(self, window=20):
        vwap = (
            self.df["volume"]
            * (self.df["high"] + self.df["low"] + self.df["close"])
            / 3
        ).cumsum() / self.df["volume"].cumsum()
        return vwap

    def calculate_macd(self, slow=26, fast=12, signal=9):
        data = self.df
        exp1 = data["close"].ewm(span=fast, adjust=False).mean()
        exp2 = data["close"].ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return macd, signal_line

    def calculate_rsi(self, window=14):
        data = self.df
        delta = data["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_vwma(self, window=20):
        v = self.df["volume"]
        tp = (self.df["high"] + self.df["low"] + self.df["close"]) / 3
        vwma = (tp * v).rolling(window).sum() / v.rolling(window).sum()
        return vwma

    def superTrend_CrossOver(self, df):
        df["ST_CROSSOVER"] = (df["vwma"] > df["SUPERT_10_4.0"]) & (
            df["vwma"].shift(1) < df["SUPERT_10_4.0"].shift(1)
        )
        return df

    def superTrend_CrossUnder(self, df):
        df["ST_CROSSUNDER"] = (df["vwma"] < df["SUPERT_10_4.0"]) & (
            df["vwma"].shift(1) > df["SUPERT_10_4.0"].shift(1)
        )
        return df

    def rsi_CrossOver(self, df, rsi_val):
        df["RSI_CROSSOVER"] = (df["rsi"] > rsi_val) & (df["rsi"].shift(1) < rsi_val)
        return df

    def rsi_CrossUnder(self, df, rsi_val):
        df["RSI_CROSSUNDER"] = (df["rsi"] < rsi_val) & (df["rsi"].shift(1) > rsi_val)
        return df
