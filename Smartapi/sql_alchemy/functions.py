from sqlalchemy.exc import SQLAlchemyError
from sql_alchemy.models import StockData

class SQLAlchemyFunctions:
    def __init__(self, session):
        self.session = session
    def add_or_update_stock_data(self,date, open, high, low, close, volume, ticker, interval):
        try:
            session = self.session
            # Check if the data already exists
            existing_data = session.query(StockData).filter_by(date=date, ticker=ticker, interval=interval).first()
            if existing_data:
                # Update existing record
                existing_data.open = open
                existing_data.high = high
                existing_data.low = low
                existing_data.close = close
                existing_data.volume = volume
                print("Updated existing data")
            else:
                # Add new record
                new_data = StockData(date=date, open=open, high=high, low=low, close=close, volume=volume, ticker=ticker, interval=interval)
                session.add(new_data)
                print("Inserted new data")
            session.commit()
        except SQLAlchemyError as e:
            session.rollback()
            print(f"Error: {e}")
