import time
from contextlib import contextmanager

import mysql.connector
from mysql.connector import Error

import root.config as config


class MySQLManager:
    def __init__(self):
        self.db_connection = self.connect_to_db()
        print(f"The database connection is: {self.db_connection}")

    def connect_to_db(self):
        retries = 3
        while retries > 0:
            try:
                conn = mysql.connector.connect(
                    database=config.DB_NAME,
                    user=config.DB_USER,
                    password=config.DB_PASSWORD,
                    host=config.DB_HOST,
                )
                if conn.is_connected():
                    print("Successfully connected to the database.")
                    return conn
            except mysql.connector.Error as e:
                print(f"Error connecting to the database: {e}")
                retries -= 1
                time.sleep(2)  # wait for 2 seconds before retrying
                if retries == 0:
                    print("Failed to connect to the database after several attempts.")
                    return None

    def close_connection(self):
        if self.db_connection and self.db_connection.is_connected():
            self.db_connection.close()
            print("Database connection closed.")

    @contextmanager
    def cursor(self):
        cur = self.db_connection.cursor()
        try:
            yield cur
        except Error as e:
            self.db_connection.rollback()
            print(f"Transaction failed: {e}")
        else:
            self.db_connection.commit()
        finally:
            cur.close()

    def execute_command(self, query, params=None):
        with self.cursor() as cur:
            cur.execute(query, params)

    def fetch_data(self, query, params=None):
        with self.cursor() as cur:
            cur.execute(query, params)
            return cur.fetchall()

    def store_data_in_db(self, data_dict, ticker):
        for interval, df in data_dict.items():
            for _, row in df.iterrows():
                try:
                    self.execute_command(
                        """
                        INSERT INTO stock_data (date, open, high, low, close, volume, ticker, interval)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        ON DUPLICATE KEY UPDATE 
                        open=VALUES(open), high=VALUES(high), low=VALUES(low), close=VALUES(close), volume=VALUES(volume)
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
                except Exception as e:
                    print(
                        f"Error inserting data for {ticker} at interval {interval}: {e}"
                    )
