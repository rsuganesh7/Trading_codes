"""_summary_
hist_data.py
"""

import datetime as dt
import time

import pandas as pd
from root.instruments_list import InstrumentDetail as ID

nifty_50 = [
    "ADANIENT",
    "ADANIPORTS",
    "APOLLOHOSP",
    "ASIANPAINT",
    "AXISBANK",
    "BAJAJ-AUTO",
    "BAJFINANCE",
    "BAJAJFINSV",
    "BPCL",
    "BHARTIARTL",
    "BRITANNIA",
    "CIPLA",
    "COALINDIA",
    "DIVISLAB",
    "DRREDDY",
    "EICHERMOT",
    "GRASIM",
    "HCLTECH",
    "HDFCBANK",
    "HDFCLIFE",
    "HEROMOTOCO",
    "HINDALCO",
    "HINDUNILVR",
    "ICICIBANK",
    "ITC",
    "INDUSINDBK",
    "INFY",
    "JSWSTEEL",
    "KOTAKBANK",
    "LTIM",
    "LT",
    "M&M",
    "MARUTI",
    "NTPC",
    "NESTLEIND",
    "ONGC",
    "POWERGRID",
    "RELIANCE",
    "SBILIFE",
    "SBIN",
    "SUNPHARMA",
    "TCS",
    "TATACONSUM",
    "TATAMOTORS",
    "TATASTEEL",
    "TECHM",
    "TITAN",
    "UPL",
    "ULTRACEMCO",
    "WIPRO",
]


class HistoricalData:
    def __init__(self, obj, exchange, duration, ticker) -> None:
        self.exchange = exchange
        self.duration = duration
        self.symbolToken = ID().token_lookup(ticker, exchange)
        self.obj = obj

    def get_dates(self, duration):
        from_date = (dt.date.today() - dt.timedelta(duration)).strftime(
            "%Y-%m-%d %H:%M"
        )
        to_date = (dt.datetime.now()).strftime("%Y-%m-%d %H:%M")
        return from_date, to_date

    def get_historical_data(self, interval):
        from_date, to_date = self.get_dates(self.duration)
        params = {
            "exchange": self.exchange,
            "symboltoken": self.symbolToken,
            "interval": interval,
            "fromdate": from_date,
            "todate": to_date,
        }
        hist_data = self.obj.getCandleData(params)
        df_data = pd.DataFrame(
            hist_data["data"],
            columns=["date", "open", "high", "low", "close", "volume"],
        )
        df_data.set_index("date", inplace=True)
        df_data.index = pd.to_datetime(df_data.index)
        df_data.sort_index(inplace=True)
        return df_data

    def hist_data_extended(self, interval):
        st_date = dt.date.today() - dt.timedelta(self.duration)
        end_date = dt.date.today()
        st_date = dt.datetime(st_date.year, st_date.month, st_date.day, 9, 15)
        end_date = dt.datetime(end_date.year, end_date.month, end_date.day)
        df_data = pd.DataFrame(
            columns=["date", "open", "high", "low", "close", "volume"]
        )

        while st_date < end_date:
            time.sleep(0.5)  # avoiding throttling rate limit
            params = {
                "exchange": self.exchange,
                "symboltoken": self.symbolToken,
                "interval": interval,
                "fromdate": (st_date).strftime("%Y-%m-%d %H:%M"),
                "todate": (end_date).strftime("%Y-%m-%d %H:%M"),
            }
            print(f"The params are: {params}")
            hist_data = self.obj.getCandleData(params)
            temp = pd.DataFrame(
                hist_data["data"],
                columns=["date", "open", "high", "low", "close", "volume"],
            )
            df_data = pd.concat([temp, df_data], ignore_index=True)

            end_date = dt.datetime.strptime(temp["date"].iloc[0][:16], "%Y-%m-%dT%H:%M")
            if (
                len(temp) <= 1
            ):  # this takes care of the edge case where start date and end date become same
                break

        df_data.set_index("date", inplace=True)
        df_data.index = pd.to_datetime(df_data.index)
        df_data.index = df_data.index.tz_localize(None)
        df_data.drop_duplicates(keep="first", inplace=True)
        return df_data

    def get_intraDay_data(self, date, interval):
        params = {
            "exchange": self.exchange,
            "symboltoken": self.symbolToken,
            "interval": interval,
            "fromdate": date.strftime("%Y-%m-%d") + "09:20",
            "todate": date.strftime("%Y-%m-%d") + "15:00",
        }
        hist_data = self.obj.getCandleData(params)
        df_data = pd.DataFrame(
            hist_data["data"],
            columns=["date", "open", "high", "low", "close", "volume"],
        )
        df_data.set_index("date", inplace=True)
        df_data.index = pd.to_datetime(df_data.index)
        df_data.index = df_data.index.tz_localize(None)
        df_data.sort_index(inplace=True)
        return df_data
