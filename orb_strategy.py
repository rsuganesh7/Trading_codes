# -*- coding: utf-8 -*-
"""
Angel One - Opening Range Breakout strategy
The code needs to be run at 9:20 IST

@author: Mayank Rasu (http://rasuquant.com/wp/)
"""

from SmartApi import SmartConnect
import os
import urllib
import json
import pandas as pd
import datetime as dt
import time
from pyotp import TOTP

starttime = time.time()
key_path = r"C:\Users\Mayank\OneDrive\Udemy\Angel One API"
os.chdir(key_path)

key_secret = open("key.txt","r").read().split()

obj=SmartConnect(api_key=key_secret[0])
data = obj.generateSession(key_secret[2],key_secret[3],TOTP(key_secret[4]).now())

instrument_url = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
response = urllib.request.urlopen(instrument_url)
instrument_list = json.loads(response.read())

tickers = ["WIPRO","ULTRACEMCO","UPL","TITAN","TECHM","TATASTEEL","TATAMOTORS","TATACONSUM","TCS","SUNPHARMA","SBIN",
           "SBILIFE","RELIANCE","POWERGRID","ONGC","NESTLEIND","NTPC","MARUTI","M&M","LT","LTIM","KOTAKBANK","JSWSTEEL",
           "INFY","INDUSINDBK","ITC","ICICIBANK","HINDUNILVR","HINDALCO","HEROMOTOCO","HDFCLIFE","HDFCBANK","HCLTECH",
           "GRASIM","EICHERMOT","DRREDDY","DIVISLAB","COALINDIA","CIPLA","BRITANNIA","BHARTIARTL","BPCL","BAJAJFINSV",
           "BAJFINANCE","BAJAJ-AUTO","AXISBANK","ASIANPAINT","APOLLOHOSP","ADANIPORTS","ADANIENT"]
pos_size = 10000
hi_lo_prices = {}

def token_lookup(ticker, instrument_list, exchange="NSE"):
    for instrument in instrument_list:
        if instrument["name"] == ticker and instrument["exch_seg"] == exchange and instrument["symbol"].split('-')[-1] == "EQ":
            return instrument["token"]
        
def symbol_lookup(token, instrument_list, exchange="NSE"):
    for instrument in instrument_list:
        if instrument["token"] == token and instrument["exch_seg"] == exchange and instrument["symbol"].split('-')[-1] == "EQ":
            return instrument["name"]
        
def get_ltp(instrument_list,ticker,exchange="NSE"):
    params = {
                "tradingsymbol":"{}-EQ".format(ticker),
                "symboltoken": token_lookup(ticker, instrument_list)
             }
    response = obj.ltpData(exchange, params["tradingsymbol"], params["symboltoken"])
    return response["data"]["ltp"]

def quantity(ticker,exchange="NSE"):
    ltp = get_ltp(instrument_list,ticker,exchange)
    return int(pos_size/ltp)
    
def get_open_orders():
    response = obj.orderBook()
    df = pd.DataFrame(response['data'])
    if len(df) > 0:
        return df[df["orderstatus"]=="open"]
    else:
        return None
    
def place_robo_order(instrument_list,ticker,buy_sell,prices,quantity,exchange="NSE"):
    ltp = get_ltp(instrument_list,ticker,exchange)
    params = {
                "variety":"ROBO",
                "tradingsymbol":"{}-EQ".format(ticker),
                "symboltoken":token_lookup(ticker, instrument_list),
                "transactiontype":buy_sell,
                "exchange":exchange,
                "ordertype":"LIMIT",
                "producttype":"BO",
                "price":ltp + 1 if buy_sell=="BUY" else ltp-1,
                "duration":"DAY",
                "stoploss": (ltp-prices[0]) if buy_sell=="BUY" else (prices[1]-ltp),
                "squareoff": round(ltp*0.05,1),                
                "quantity":quantity
                }
    response = obj.placeOrder(params)
    return response
        
def hist_data_0920(tickers,duration,interval,instrument_list,exchange="NSE"):
    hist_data_tickers = {}
    for ticker in tickers:
        time.sleep(0.4)
        params = {
                 "exchange": exchange,
                 "symboltoken": token_lookup(ticker,instrument_list),
                 "interval": interval,
                 "fromdate": (dt.date.today() - dt.timedelta(duration)).strftime('%Y-%m-%d %H:%M'),
                 "todate": dt.date.today().strftime('%Y-%m-%d') + ' 09:19' #there seems to be a bug in smartAPI's historical data call as it also providing candles starting from to_data
                 }
        hist_data = obj.getCandleData(params)
        df_data = pd.DataFrame(hist_data["data"],
                               columns = ["date","open","high","low","close","volume"])
        df_data.set_index("date",inplace=True)
        df_data.index = pd.to_datetime(df_data.index)
        df_data.index = df_data.index.tz_localize(None)
        df_data["gap"] = ((df_data["open"]/df_data["close"].shift(1))-1)*100
        hist_data_tickers[ticker] = df_data
    return hist_data_tickers




def filtered_tickers(data):
    date = data[tickers[0]].index.to_list()[-1] 
    temp = pd.Series()
    for ticker in data:
        temp.loc[ticker] = data[ticker].loc[date,"gap"]
    return (abs(temp[abs(temp)>1.5])).sort_values(ascending=False)[:5].index.to_list()

def orb_strat(tickers,hi_lo_prices,exchange="NSE"):
    positions = pd.DataFrame(obj.position()["data"])
    open_orders = get_open_orders()
    if len(positions) > 0:
        tickers = [i for i in tickers if i+"-EQ" not in positions["tradingsymbol"].to_list()]
    if open_orders is not None:
        tickers = [i for i in tickers if i+"-EQ" not in open_orders["tradingsymbol"].to_list()]
    for ticker in tickers:
        time.sleep(0.4)
        params = {
                 "exchange": exchange,
                 "symboltoken": token_lookup(ticker,instrument_list),
                 "interval": "FIVE_MINUTE",
                 "fromdate": (dt.date.today() - dt.timedelta(6)).strftime('%Y-%m-%d %H:%M'),
                 "todate": dt.datetime.now().strftime('%Y-%m-%d %H:%M')
                 }
        hist_data = obj.getCandleData(params)
        df_data = pd.DataFrame(hist_data["data"],
                               columns = ["date","open","high","low","close","volume"])
        df_data.set_index("date",inplace=True)
        df_data.index = pd.to_datetime(df_data.index)
        df_data.index = df_data.index.tz_localize(None)
        df_data["avvol"] = df_data["volume"].rolling(10).mean().shift(1)
        
        
        if df_data["volume"].iloc[-1] >= 2*df_data["avvol"].iloc[-1]:
            if df_data["high"].iloc[-1] >= hi_lo_prices[ticker][1] and df_data["low"].iloc[-1] >= hi_lo_prices[ticker][0]:
                place_robo_order(instrument_list, ticker, "BUY", hi_lo_prices[ticker], quantity(ticker))
                print("bought {} stocks of {}".format(quantity(ticker),ticker))
            elif df_data["low"].iloc[-1] <= hi_lo_prices[ticker][0] and df_data["high"].iloc[-1] <= hi_lo_prices[ticker][1]:
                place_robo_order(instrument_list, ticker, "SELL", hi_lo_prices[ticker], quantity(ticker))
                print("sold {} stocks of {}".format(quantity(ticker),ticker))
            
#extract the historical data at 9:20 am         
data_0920 = hist_data_0920(tickers, 4, "FIVE_MINUTE", instrument_list)
tickers = filtered_tickers(data_0920) #identify tickers with maximum gap up/down
print("filtered tickers")
print(tickers)
for ticker in tickers:
    hi_lo_prices[ticker] = [data_0920[ticker]["low"].iloc[-1],data_0920[ticker]["high"].iloc[-1]]
time.sleep(300 - ((time.time() - starttime) % 300.0))

#run the code till 2:30 pm
while dt.datetime.now() < dt.datetime.strptime(dt.datetime.now().strftime('%Y-%m-%d')+' 14:30','%Y-%m-%d %H:%M'):
    print("starting passthrough at {}".format(dt.datetime.now()))
    orb_strat(tickers,hi_lo_prices)
    time.sleep(300 - ((time.time() - starttime) % 300.0))        

