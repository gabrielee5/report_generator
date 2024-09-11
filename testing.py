from pybit.unified_trading import HTTP
from dotenv import dotenv_values
import datetime
from decimal import Decimal
from main import parse_executions

# Load environment variables
secrets = dotenv_values(".env")
api_key = secrets["001_api_key"]
api_secret = secrets["001_api_secret"]

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

# Fetch executions
response = session.get_executions(
    category="linear",
    limit=100,
)
# print(response)
# print(get_all_trades_today(session))


'''
# Display executions
if response['retCode'] == 0:
    display_executions(response['result']['list'])
else:
    print(f"Error: {response['retMsg']}")
'''

def equity_btc(session, equity_usdt):
    response = session.get_tickers(
    category="spot",
    symbol="BTCUSDT",
    )
    last_price_btc = float(response["result"]["list"][0]["lastPrice"])
    return equity_usdt / last_price_btc

# print(equity_btc(session, 10000))

withdrawal = session.get_withdrawal_records(
    coin="USDT",
    withdrawType=2,
    limit=10,
)

deposit = session.get_internal_deposit_records(
    # startTime=1724168100000,
    # endTime=1725982560000,
)

transfer = session.get_internal_transfer_records(
    coin="USDT",
    limit=3,
)

deposit2 = session.get_deposit_records(
    coin="USDT",
)

# print(transfer)
print(deposit2)
# print(withdrawal)


