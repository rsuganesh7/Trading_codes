# In[0]

import yfinance as yf

# In[]
# Download historical data for a stock/index (e.g., ^NSEI. - Nifty 50)
ticker = "RELIANCE.NS"
data = yf.download(tickers=ticker, period="50d", interval="5m")


# %%
data
# In[]
import datetime

today = datetime.date.today()

look_back = 500

end_date = today.strftime("%Y-%m-%d")
start_date = (today - datetime.timedelta(days=look_back)).strftime("%Y-%m-%d")
