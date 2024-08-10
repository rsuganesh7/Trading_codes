import logging
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import yfinance as yf
from Firebase import InitializeFirebase


class HistData:
    def __init__(self, ticker):
        self.ticker = ticker
        self.cleaned_ticker = self.sanitize_ticker(self.ticker)
        logging.info(f"Fetching data for {self.ticker}")

    def fetch_and_upload(self, db):
        try:
            data = yf.download(self.ticker, period="60d", interval="5m")
            if not data.empty:
                self.upload_data(data, db)
            else:
                logging.warning(f"No data fetched for {self.ticker}")
        except Exception as e:
            logging.error(f"Failed to fetch data for {self.ticker}: {e}")

    def upload_data(self, data, db):
        try:
            stock_ref = db.child("stocks").child(self.cleaned_ticker)
            for index, row in data.iterrows():
                date_str = index.strftime("%Y-%m-%d %H:%M:%S")
                stock_ref.child(date_str).set(row.to_dict())
            logging.info(f"Data uploaded for {self.ticker}")
        except Exception as e:
            logging.error(f"Failed to upload data for {self.ticker}: {e}")

    def sanitize_ticker(self, ticker):
        return ticker.replace(".", "_")  # Replace periods with underscores


class Main:
    def __init__(self):
        self.firebase = InitializeFirebase()
        self.db = self.firebase.db
        self.nifty_200 = pd.read_csv("ind_nifty200list.csv")["Symbol"].tolist()

    def process_stock(self, ticker):
        symbol = str(ticker) + ".NS"
        hist_data = HistData(symbol)
        hist_data.fetch_and_upload(self.db)
        logging.info(f"Data uploaded for {ticker}")

    def run(self):
        with ThreadPoolExecutor(
            max_workers=10
        ) as executor:  # Adjust number of workers as needed
            executor.map(self.process_stock, self.nifty_200)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main_process = Main()
    main_process.run()
