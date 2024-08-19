# main.py

import os
import pandas as pd
from root.hist_data import HistoricalData
from root.login_manager import LoginManager
import pandas as pd


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
        nifty_200 = pd.read_csv("FnO.csv")["SYMBOL"].to_list()
        
        # Ensure the directory exists
        output_path ="FnO/5M"
        os.makedirs(output_path, exist_ok=True)
        
        for ticker in nifty_200:
            try:
                data_dict = self.prepare_data(
                    self.obj, "NSE", 1500, ticker, "FIVE_MINUTE"
                )
                print(f'Data for {ticker} is being processed...')
                
                # Creating DataFrame in a more concise manner
                modified_dict = pd.DataFrame({
                    'Datetime': data_dict.index,
                    'Open': data_dict['open'],
                    'High': data_dict['high'],
                    'Low': data_dict['low'],
                    'Close': data_dict['close'],
                    'Volume': data_dict['volume']
                })
                
                # Saving the DataFrame to a CSV file
                modified_dict.to_csv(f"{output_path}/{ticker}.csv", index=False)
                print(f'Data for {ticker} saved successfully.')
            
            except Exception as e:
                print(f"Error processing data for {ticker}: {e}")
                # Optionally log errors to a file
                with open("error_log.txt", "a") as log_file:
                    log_file.write(f"Error processing data for {ticker}: {e}\n")


    def prepare_data(self, obj, exchange, duration, ticker, interval):
        SYM_DATA = HistoricalData(obj, exchange, duration, ticker)
        df = SYM_DATA.hist_data_extended(interval)
        return df


if __name__ == "__main__":
    Main()
