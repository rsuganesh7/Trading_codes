# main.py

import pandas as pd
from root.hist_data import HistoricalData
from root.login_manager import LoginManager
import pandas as pd

from root.sql_alchemy import SQLAlchemyManager
from sql_alchemy.functions import SQLAlchemyFunctions

class Main:
    def __init__(self):
        self.login_manager = LoginManager()
        self.obj, self.feed_token, self.user_profile = None, None, None
        self.initialize()

    def initialize(self):
        try:
            self.obj, self.feed_token, self.user_profile = (
                self.login_manager.login_and_retrieve_info()
            )
            self.process_stocks()
        except Exception as e:
            print(f"Failed to login or retrieve stock data: {e}")

    def process_stocks(self):
        nifty_200 = pd.read_csv("ind_nifty200list.csv")["Symbol"].to_list()
        for ticker in nifty_200:
            try:
                data_dict = self.prepare_data(
                    self.obj, "NSE", 200, ticker, "FIVE_MINUTE"
                )
                data_dict.to_csv(f"nifty_200/data/{ticker}.csv")
            except Exception as e:
                print(f"Error processing data for {ticker}: {e}")

    


    def prepare_data(self, obj, exchange, duration, ticker, interval):
        SYM_DATA = HistoricalData(obj, exchange, duration, ticker)
        df = SYM_DATA.get_historical_data(interval)
        return df

    def __del__(self):
        if self.db_manager:
            self.db_manager.close_connection()


if __name__ == "__main__":
    Main()
