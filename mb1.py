import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd

# List of stock symbols
stocks = [
    'MOREPENLAB.NS', 'ELECTCAST.NS', 'STAR.NS', 'KARURVYSYA.NS', 'HFCL.NS',
    'LIBERTSHOE.NS', 'CASTROLIND.NS', 'BBOX.NS', 'MARKSANS.NS', 'ELECTHERM.NS',
    'TIDEWATER.NS', 'FORTIS.NS', 'GLOBALVECT.NS', 'SMSPHARMA.NS', 'INSPIRISYS.NS',
    'WALCHANNAG.NS', 'GMRINFRA.NS', 'SUZLON.NS', 'HCC.NS', 'RKDL.NS', 
    'BAJAJHIND.NS', 'INOXWIND.NS', 'INDUSTOWER.NS', 'THEMISMED.NS', 'COUNCODOS.NS',
    'SARLAPOLY.NS', 'SHILPAMED.NS', 'ELGIRUBCO.NS', 'EVEREADY.NS', 'ASAHISONG.NS',
    'CAPACITE.NS', 'TPLPLASTEH.NS', 'CHEMFAB.NS', 'ATLANTAA.NS', 'KHADIM.NS',
    'GRANULES.NS', 'CAMS.NS', 'FIBERWEB.NS', 'TCPLPACK.NS', 'RITCO.NS', 
    'SKIPPER.NS', 'NATCOPHARM.NS', 'IFBIND.NS', 'ERIS.NS', 'HINDPETRO.NS', 
    'ARROWGREEN.NS', 'NAHARSPING.NS', 'VETO.NS', 'OPTIEMUS.NS', 'SANDHAR.NS',
    'SUMMITSEC.NS', 'SUNTV.NS', '21STCENMGM.NS', 'ALANKIT.NS', 'SUVEN.NS', 'OMAXE.NS'
]

# Fetch data and plot
for stock in stocks:
    data = yf.download(stock, period='max')
    if not data.empty:
        plt.figure(figsize=(12, 6))
        plt.plot(data.index, data['Adj Close'], label=stock)
        plt.title(f'{stock} Price Trend')
        plt.xlabel('Date')
        plt.ylabel('Adjusted Close Price')
        plt.legend()
        plt.grid(True)
        plt.show()
    else:
        print(f"Data for {stock} not available or could not be retrieved.")
