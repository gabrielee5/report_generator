import ccxt
from dotenv import dotenv_values
import os
import logging

def create_bybit_connection():
    exchange_class = ccxt.bybit

    # Load environment variables
    secrets = dotenv_values(".env")
    api_key = secrets["api_key"]
    api_secret = secrets["api_secret"]

    exchange_params = {
        'apiKey': api_key,
        'secret': api_secret,
        'enableRateLimit': True,
    }

    exchange = exchange_class(exchange_params)

    return exchange

def test_connection(exchange):
    try:
        # Fetch the balance to test the connection
        balance = exchange.fetch_balance()
        print("Connection successful!")
        print(f"Total USDT balance: {balance['total']['USDT']}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

# added to test functions
def fetch_account_data(exchange):
    balance = exchange.fetch_balance()
    prova = exchange.fetchBalance()
    portfolio_value = balance['total']['USDT']
    positions = exchange.fetchPositions()
    print(f"Portfolio Value: ${portfolio_value:.2f}")
    print(f"Portfolio Positions: {positions}")
    # print(prova)

if __name__ == "__main__":
    bybit = create_bybit_connection()
    test_connection(bybit)
    # fetch_account_data(bybit)