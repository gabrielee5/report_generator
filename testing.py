from pybit.unified_trading import HTTP
from dotenv import dotenv_values
import datetime
from decimal import Decimal
from main import parse_executions
import json


# Load environment variables
secrets = dotenv_values(".env")
api_key = secrets["003_api_key"]
api_secret = secrets["003_api_secret"]

# Initialize Bybit client
session = HTTP(
    api_key=api_key,
    api_secret=api_secret
)

def display_executions(executions):
    print("\nExecution Data:")
    print("-" * 100)
    print(f"{'Symbol':<10} {'Side':<5} {'Price':<10} {'Quantity':<10} {'Value':<12} {'Type':<8} {'Time':<25}")
    print("-" * 100)
    
    for exec in executions:
        symbol = exec['symbol']
        side = exec['side']
        price = float(exec['execPrice'])
        qty = float(exec['execQty'])
        value = float(exec['execValue'])
        exec_type = exec['execType']
        
        print(f"{symbol:<10} {side:<5} {price:<10.4f} {qty:<10.4f} {value:<12.2f} {exec_type:<8}")

def get_all_trades_today(session):
    today = datetime.datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    all_trades = []
    cursor = None

    response = session.get_executions(
                category="linear",
                limit=100,
                cursor=cursor
            )

    executions = response["result"]["list"]
    parsed_executions = parse_executions(executions, today)
    all_trades.extend(parsed_executions)

    return all_trades

def process_fees(data, current_date):
    funding_fees = Decimal('0')
    trading_fees = Decimal('0')
    
    # Convert current_date to datetime object for comparison
    current_date = datetime.datetime.strptime(current_date, "%Y-%m-%d").date()

    for execution in data['result']['list']:
        exec_time = datetime.datetime.fromtimestamp(int(execution['execTime']) / 1000).date()
        
        # Check if the execution is from the current day
        if exec_time == current_date:
            fee = Decimal(execution['execFee'])
            
            if execution['execType'] == 'Funding':
                funding_fees += fee
            elif execution['execType'] == 'Trade':
                trading_fees += fee

    return {
        'funding_fees': round(funding_fees, 8),
        'trading_fees': round(trading_fees, 8),
        'total_fees': round(funding_fees + trading_fees, 8)
    }


def equity_btc(session, equity_usdt):
    response = session.get_tickers(
    category="spot",
    symbol="BTCUSDT",
    )
    last_price_btc = float(response["result"]["list"][0]["lastPrice"])
    return equity_usdt / last_price_btc

# print(equity_btc(session, 10000))


def filter_coin_data(data, coin_symbol):
    # Check if data is already a dictionary
    if isinstance(data, dict):
        parsed_data = data
    else:
        try:
            # If it's a string, try to parse it as JSON
            parsed_data = json.loads(data)
        except json.JSONDecodeError:
            return "Error: Invalid data format. Expected a dictionary or valid JSON string."
    
    # Extract the list of trades
    trades = parsed_data.get('result', {}).get('list', [])
    
    # Filter trades for the specified coin
    filtered_trades = [trade for trade in trades if trade['symbol'] == coin_symbol]
    
    # If no trades found for the specified coin, return a message
    if not filtered_trades:
        return f"No trades found for {coin_symbol}"
    
    # Create a formatted output
    output = f"Trades for {coin_symbol}:\n"
    for trade in filtered_trades:
        output += f"Time: {trade['execTime']}\n"
        output += f"Side: {trade['side']}\n"
        output += f"Price: {trade['execPrice']}\n"
        output += f"Quantity: {trade['execQty']}\n"
        output += f"Value: {trade['execValue']}\n"
        output += f"Fee: {trade['execFee']}\n"
        output += f"Type: {trade['execType']}\n"
        output += "-" * 30 + "\n"
    
    return output

# Fetch executions
output = session.get_executions(
    category="linear",
    limit=100,
)

response = session.get_positions(
            category="linear",
            settleCoin="USDT"
        )
print(response)