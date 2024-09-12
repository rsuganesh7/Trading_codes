import firebase_admin
from firebase_admin import credentials, firestore
import os
import pandas as pd
from root.hist_data import HistoricalData
from root.login_manager import LoginManager
import math

class Main:
    def __init__(self):
        # Initialize Firebase only if it hasn't been initialized already
        if not firebase_admin._apps:
            cred = credentials.Certificate("Trading_codes/rare-style-435014-p3-cb0d67622335.json")
            firebase_admin.initialize_app(cred)
        
        self.db = firestore.client()
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

    def prepare_data(self, obj, exchange, duration, ticker, interval):
        SYM_DATA = HistoricalData(obj, exchange, duration, ticker)
        df = SYM_DATA.hist_data_extended(interval)
        return df

    def process_stocks(self):
        nifty_200 = pd.read_csv("/Users/suganeshr/Trading/Trading_codes/FnO.csv")["SYMBOL"].to_list()
        
        intervals = ['FIVE_MINUTE', 'ONE_HOUR']
        max_docs_per_batch = 500  # Limit documents per batch (Firestore limit is 500 writes per batch)
        
        for interval in intervals:
            for ticker in nifty_200:
                try:
                    data_dict = self.prepare_data(
                        self.obj, "NSE", 1500, ticker, interval
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

                    # Split data into chunks if it exceeds batch write limit
                    chunk_size = max_docs_per_batch
                    num_chunks = math.ceil(len(modified_dict) / chunk_size)

                    for i in range(num_chunks):
                        batch = self.db.batch()
                        chunk = modified_dict.iloc[i*chunk_size:(i+1)*chunk_size]

                        for _, row in chunk.iterrows():
                            # Check if the row has no missing data
                            if not row.isnull().values.any():
                                doc_ref = self.db.collection('stocks').document(ticker).collection(interval).document(str(row['Datetime']))
                                batch.set(doc_ref, row.to_dict())

                        # Commit the batch write to Firestore
                        batch.commit()
                        print(f"Batch {i+1} for {ticker} ({interval}) saved successfully.")

                except Exception as e:
                    print(f"Error processing data for {ticker}: {e}")
                    # Optionally log errors to a file
                    with open("error_log.txt", "a") as log_file:
                        log_file.write(f"Error processing data for {ticker}: {e}\n")


if __name__ == "__main__":
    Main()
