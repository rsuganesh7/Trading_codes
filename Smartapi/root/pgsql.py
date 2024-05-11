import os

import psycopg2


class PostGresManager:
    def __init__(self):
        self.db_connection = self.connect_to_db()
        print(f"The database connection is: {self.db_connection}")

    def connect_to_db(self):
        try:
            conn = psycopg2.connect(
                dbname=os.getenv("DB_NAME", "stock_data"),
                user=os.getenv("DB_USER", "trader"),
                password=os.getenv("DB_PASSWORD", "your_default_password"),
                host=os.getenv("DB_HOST", "localhost"),
            )
            return conn
        except psycopg2.Error as e:
            print(f"Error connecting to the database: {e}")
            return None

    def close_connection(self):
        if self.db_connection:
            self.db_connection.close()
            print("Database connection closed.")

    def execute_query(self, query, params=None):
        with self.db_connection.cursor() as cursor:
            try:
                cursor.execute(query, params)
                self.db_connection.commit()
                return cursor.fetchall()
            except psycopg2.Error as e:
                self.db_connection.rollback()
                print(f"Error executing query: {e}")
                return None

    def store_data_in_db(self, data_dict, ticker):
        cursor = self.db_connection.cursor()
        for interval, df in data_dict.items():
            for _, row in df.iterrows():
                try:
                    cursor.execute(
                        """
                        INSERT INTO stock_data (date, open, high, low, close, volume, ticker, interval)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (date, ticker, interval) DO NOTHING
                        """,
                        (
                            row.name,
                            row["open"],
                            row["high"],
                            row["low"],
                            row["close"],
                            row["volume"],
                            ticker,
                            interval,
                        ),
                    )
                    self.db_connection.commit()
                except psycopg2.Error as e:
                    print(
                        f"Error inserting data for {ticker} at interval {interval}: {e}"
                    )
                    self.db_connection.rollback()
        cursor.close()
