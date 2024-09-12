import mysql.connector
from mysql.connector import Error

def connect_to_db():
    try:
        connection = mysql.connector.connect(
            host='localhost',
            database='stock_data',
            user='root',
            password='sugan0797'
        )
        if connection.is_connected():
            return connection
    except Error as e:
        print(f"Error while connecting to MySQL: {e}")
        return None

def create_table_if_not_exists(cursor, table_name):
    create_table_query = f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
        id INT AUTO_INCREMENT PRIMARY KEY,
        rsi_period INT,
        vwma_length INT,
        stop_loss FLOAT,
        take_profit FLOAT,
        buy_date DATETIME,
        buy_price FLOAT,
        sell_date DATETIME,
        sell_price FLOAT,
        quantity INT,
        returns FLOAT,
        pnl FLOAT,
        realized_profit FLOAT,
        cum_profit FLOAT,
        cumulative_return FLOAT,
        drawdown FLOAT
    );
    """
    cursor.execute(create_table_query)
    cursor.fetchall()  # Ensure the result is fully processed (if applicable)

def insert_backtest_results(cursor, table_name, rsi_period, vwma_length, stop_loss, take_profit, trade_history_df):
    insert_query = f"""
    INSERT INTO {table_name} (rsi_period, vwma_length, stop_loss, take_profit, buy_date, buy_price, sell_date, sell_price, quantity, returns, pnl, realized_profit, cum_profit, cumulative_return, drawdown)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    # Batch insert data
    batch_data = []
    for _, row in trade_history_df.iterrows():
        batch_data.append((
            rsi_period, vwma_length, stop_loss, take_profit,
            row['Buy Date'], row['Buy Price'], row['Sell Date'], row['Sell Price'], row['Quantity'],
            row['Return'], row['PnL'], row['Realized Profit'], row['Cum Profit'], row['Cumulative Return'], row['Drawdown']
        ))
    cursor.executemany(insert_query, batch_data)
    cursor.fetchall()  # Ensure the result is fully processed (if applicable)

