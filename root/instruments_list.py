"""_summary_
instrument_list.py
"""

import json
import urllib
import urllib.request


class InstrumentDetail:
    def __init__(self):
        self.nifty_50 = [
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

    # def get_instrument_list(self):
    #     url = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
    #     response = urllib.request.urlopen(url)
    #     data = json.loads(response.read())
    #     return data
    def get_instrument_list(self):
        with open('/Users/suganeshr/Trading/Trading_codes/il.json', 'r') as f:
            il = json.load(f)
        return il

    def token_lookup(self, ticker, exchange="NSE"):
        for instrument in self.get_instrument_list():
            if (
                instrument["name"] == ticker
                and instrument["exch_seg"] == exchange
                and instrument["symbol"].split("-")[-1] == "EQ"
            ):
                return instrument["token"]

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
