from pybit.unified_trading import HTTP
from dotenv import dotenv_values
import datetime
from pybit.exceptions import InvalidRequestError
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, landscape
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor
from reportlab.platypus import Image, PageBreak
import sqlite3
import os
import csv
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from io import BytesIO
from decimal import Decimal
import logging
import time
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import colorsys
import argparse

os.chdir(os.path.dirname(os.path.abspath(__file__)))

db_path = 'db/database.db'

# Set up logging
logging.basicConfig(filename='trading_report.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def get_accounts_from_env():
    try:
        env_vars = dotenv_values(".env")
        accounts = []
        
        sorted_keys = sorted(env_vars.keys())
        
        for key in sorted_keys:
            if key.endswith('_api_key'):
                account_id = key[:-8]
                accounts.append({
                    "id": account_id,
                    "name": env_vars.get(f"{account_id}_name", f"Account {account_id}"),
                    "api_key": env_vars[f"{account_id}_api_key"],
                    "api_secret": env_vars[f"{account_id}_api_secret"]
                })
        
        if not accounts:
            raise ValueError("No accounts found in .env file")
        
        return accounts
    except Exception as e:
        logging.error(f"Error reading accounts from .env: {str(e)}")
        raise
    
# DATABASE
def initialize_database(db_path=db_path):
    """
    Initialize the database with the daily_reports table and required columns.
    
    Args:
        db_path (str): Path to the database file
        
    Returns:
        bool: True if initialization successful, False otherwise
    """
    conn = None
    try:
        # Check if this is a new database
        is_new_db = not os.path.exists(db_path)
        
        # Connect with foreign key support enabled
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA foreign_keys = ON")
        cursor = conn.cursor()

        # Enable Write-Ahead Logging for better concurrency
        cursor.execute("PRAGMA journal_mode=WAL")
        
        # Create the main table with proper constraints and types
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS daily_reports (
            date TEXT NOT NULL CHECK (date IS date(date, '+0 days')),
            account_name TEXT NOT NULL CHECK (length(account_name) > 0),
            equity REAL NOT NULL CHECK (equity >= 0),
            open_positions INTEGER NOT NULL DEFAULT 0 CHECK (open_positions >= 0),
            trades_today INTEGER NOT NULL DEFAULT 0 CHECK (trades_today >= 0),
            long_positions INTEGER NOT NULL DEFAULT 0 CHECK (long_positions >= 0),
            short_positions INTEGER NOT NULL DEFAULT 0 CHECK (short_positions >= 0),
            long_exposure REAL NOT NULL DEFAULT 0,
            short_exposure REAL NOT NULL DEFAULT 0,
            funding_fees REAL NOT NULL DEFAULT 0,
            trading_fees REAL NOT NULL DEFAULT 0,
            total_volume REAL NOT NULL DEFAULT 0 CHECK (total_volume >= 0),
            deposit REAL NOT NULL DEFAULT 0 CHECK (deposit >= 0),
            withdrawal REAL NOT NULL DEFAULT 0 CHECK (withdrawal >= 0),
            PRIMARY KEY (date, account_name)
        )
        ''')

        # Create indexes for better query performance
        cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_daily_reports_account 
        ON daily_reports(account_name)
        ''')
        
        cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_daily_reports_date 
        ON daily_reports(date)
        ''')

        # Verify/add required columns
        cursor.execute("PRAGMA table_info(daily_reports)")
        existing_columns = {column[1] for column in cursor.fetchall()}
        
        required_columns = {
            'deposit': 'REAL NOT NULL DEFAULT 0 CHECK (deposit >= 0)',
            'withdrawal': 'REAL NOT NULL DEFAULT 0 CHECK (withdrawal >= 0)'
        }
        
        for column, definition in required_columns.items():
            if column not in existing_columns:
                cursor.execute(f"ALTER TABLE daily_reports ADD COLUMN {column} {definition}")
                logging.info(f"Added missing column: {column}")

        # Verify table structure
        cursor.execute("PRAGMA table_info(daily_reports)")
        columns = cursor.fetchall()
        if len(columns) != 14:  # Expected number of columns
            logging.warning(f"Unexpected number of columns: {len(columns)}")

        conn.commit()
        logging.info(f"{'Created new' if is_new_db else 'Connected to existing'} database: {db_path}")
        return True

    except sqlite3.Error as e:
        logging.error(f"SQLite error during initialization: {str(e)}")
        if conn:
            conn.rollback()
        raise
    except Exception as e:
        logging.error(f"Unexpected error during initialization: {str(e)}")
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            try:
                conn.close()
            except Exception as e:
                logging.error(f"Error closing database connection: {str(e)}")

# DATA
def store_daily_data(data):
    """
    Store daily trading report data in the database, handling both new records and updates.
    
    Args:
        report_data (dict): Trading report data containing account metrics
    """
    conn = None
    try:
        # Convert all Decimal values to float
        processed_data = {
            key: float(value) if isinstance(value, Decimal) else value 
            for key, value in data.items()
        }
        
        # Prepare the common fields for both insert and update
        data_fields = (
            processed_data['equity'],
            processed_data['open_positions'],
            processed_data['trades_today'],
            processed_data['long_positions'],
            processed_data['short_positions'],
            processed_data['long_exposure'],
            processed_data['short_exposure'],
            processed_data['funding_fees'],
            processed_data['trading_fees'],
            processed_data['total_volume']
        )

        with sqlite3.connect(db_path) as conn:  # Changed to database2.db
            cursor = conn.cursor()
            
            # Check for existing record
            cursor.execute('''
                SELECT 1 FROM daily_reports
                WHERE date = ? AND account_name = ?
            ''', (processed_data['date'], processed_data['account_name']))
            
            record_exists = cursor.fetchone() is not None

            if record_exists:
                # Update existing record, preserving deposit and withdrawal
                cursor.execute('''
                    UPDATE daily_reports 
                    SET equity = ?,
                        open_positions = ?,
                        trades_today = ?,
                        long_positions = ?,
                        short_positions = ?,
                        long_exposure = ?,
                        short_exposure = ?,
                        funding_fees = ?,
                        trading_fees = ?,
                        total_volume = ?
                    WHERE date = ? AND account_name = ?
                ''', data_fields + (processed_data['date'], processed_data['account_name']))
            else:
                # Insert new record
                cursor.execute('''
                    INSERT INTO daily_reports (
                        date, account_name, equity, open_positions,
                        trades_today, long_positions, short_positions, long_exposure,
                        short_exposure, funding_fees, trading_fees,
                        total_volume, deposit, withdrawal
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    processed_data['date'],
                    processed_data['account_name']
                ) + data_fields + (
                    processed_data.get('deposit', 0),
                    processed_data.get('withdrawal', 0)
                ))

            rows_affected = cursor.rowcount
            logging.info(
                f"{'Updated' if record_exists else 'Inserted'} record for "
                f"{processed_data['account_name']} on {processed_data['date']}. "
                f"Rows affected: {rows_affected}"
            )

    except sqlite3.Error as e:
        logging.error(f"Database error while storing daily report: {str(e)}")
        raise
    except KeyError as e:
        logging.error(f"Missing required field in report data: {str(e)}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error while storing daily report: {str(e)}")
        raise

def get_existing_transaction_data(date, account_name):
    """
    Get and update transaction data from both database and CSV file.
    
    Args:
        date (str): Date in YYYY-MM-DD format
        account_name (str): Name of the account
        
    Returns:
        tuple: (deposit amount, withdrawal amount) for the specified date
    """
    try:
        # First, check the CSV file for any transactions
        csv_transactions = {}
        try:
            with open('transactions.csv', 'r') as csvfile:
                csvreader = csv.DictReader(csvfile)
                for row in csvreader:
                    if row['Account Name'].lower() == account_name.lower():  # Case-insensitive comparison
                        transaction_date = row['Timestamp']
                        amount = float(row['Amount'])
                        transaction_type = row['Type'].lower()
                        
                        # Initialize or update the transaction data for this date
                        if transaction_date not in csv_transactions:
                            csv_transactions[transaction_date] = {'deposit': 0, 'withdrawal': 0}
                        
                        # Add the transaction amount
                        if transaction_type == 'deposit':
                            csv_transactions[transaction_date]['deposit'] += amount
                        elif transaction_type == 'withdrawal':
                            csv_transactions[transaction_date]['withdrawal'] += amount
                        
                        logging.info(f"Found {transaction_type} of {amount} for {account_name} on {transaction_date}")
        
        except FileNotFoundError:
            logging.warning("transactions.csv not found")
        
        # Connect to database and update/retrieve data
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            # Update historical data if needed
            for trans_date, amounts in csv_transactions.items():
                cursor.execute('''
                    SELECT deposit, withdrawal 
                    FROM daily_reports 
                    WHERE date = ? AND account_name = ?
                ''', (trans_date, account_name))
                
                db_record = cursor.fetchone()
                
                if db_record:
                    # If amounts differ, update the database
                    if abs(db_record[0] - amounts['deposit']) > 0.01 or abs(db_record[1] - amounts['withdrawal']) > 0.01:
                        cursor.execute('''
                            UPDATE daily_reports 
                            SET deposit = ?, withdrawal = ?
                            WHERE date = ? AND account_name = ?
                        ''', (amounts['deposit'], amounts['withdrawal'], trans_date, account_name))
                        logging.info(f"Updated transactions for {account_name} on {trans_date}")
            
            # Commit any updates
            conn.commit()
            
            # Get the requested date's data
            cursor.execute('''
                SELECT deposit, withdrawal 
                FROM daily_reports 
                WHERE date = ? AND account_name = ?
            ''', (date, account_name))
            
            db_data = cursor.fetchone()
            
            # Return data based on priority (CSV > DB > defaults)
            if date in csv_transactions:
                return csv_transactions[date]['deposit'], csv_transactions[date]['withdrawal']
            elif db_data:
                return db_data[0], db_data[1]
            else:
                return 0, 0
                
    except sqlite3.Error as e:
        logging.error(f"Database error while getting transaction data: {str(e)}")
        raise
    except Exception as e:
        logging.error(f"Error processing transaction data: {str(e)}")
        raise

def get_all_open_positions(session):
    try:
        response = session.get_positions(
            category="linear",
            settleCoin="USDT"
        )
        positions = response["result"]["list"]
        return [position for position in positions if float(position["size"]) != 0]
    except InvalidRequestError as e:
        logging.error(f"Error getting open positions: {str(e)}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error getting open positions: {str(e)}")
        raise

def parse_executions(executions, start_time):
    return [
        {
            "symbol": execution["symbol"],
            "side": execution["side"],
            "price": float(execution["execPrice"]),
            "quantity": float(execution["execQty"]),
            "value": float(execution["execValue"]),
            "fee": float(execution["execFee"]),
            "time": datetime.datetime.fromtimestamp(int(execution["execTime"]) / 1000),
            "type": execution["execType"],
            "orderId": execution["orderId"]
        }
        for execution in executions
        if execution["execType"] == "Trade" and int(execution["execTime"]) >= int(start_time.timestamp() * 1000)
    ]

def last_executions(session):
    cursor = None
    response = session.get_executions(
        category="linear",
        limit=100,
        cursor=cursor
        )
    return response

def get_all_trades_today(session):
    today = datetime.datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    all_trades = []
    cursor = None

    try:
        while True:
            response = last_executions(session)
            executions = response["result"]["list"]
            parsed_executions = parse_executions(executions, today)
            all_trades.extend(parsed_executions)

            cursor = response["result"].get("nextPageCursor")
            if not cursor or not executions or (executions and int(executions[-1]["execTime"]) < int(today.timestamp() * 1000)):
                break

        return all_trades
    except InvalidRequestError as e:
        logging.error(f"Error fetching executions: {str(e)}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error fetching trades: {str(e)}")
        raise

def format_position_data(position):
    """Format a single position for the report."""
    return {
        "symbol": position["symbol"],
        "side": position["side"],
        "exposure": float(position["size"]) * float(position["markPrice"]),
        "entry_price": float(position["avgPrice"]),
        "mark_price": float(position["markPrice"]),
        "unrealized_pnl": float(position["unrealisedPnl"]),
        "leverage": position["leverage"]
    }

def determine_trade_action(trades, open_positions):
    # Sort trades by time to ensure chronological order
    sorted_trades = sorted(trades, key=lambda x: x['time'])
    
    # Initialize positions from open_positions
    positions = {
        position["symbol"]: {
            "size": float(position["size"]),
            "side": "Long" if position["side"] == "Buy" else "Short"
        }
        for position in open_positions
    }

    processed_trades = []

    for trade in sorted_trades:
        symbol = trade["symbol"]
        side = trade["side"]
        
        if symbol not in positions:
            positions[symbol] = {"size": 0, "side": None}
        
        now_position = positions[symbol]["size"]
    
        # Determine action based on how the position changed
        if now_position == 0:
            action = f"Closing {'Long' if side == 'Sell' else 'Short'}"
        elif now_position > 0:
            action = f"Opening {'Long' if side == 'Buy' else 'Short'}"
        
        positions[symbol] = {"size": now_position, "side": "Long" if now_position > 0 else "Short"}
        
        processed_trade = trade.copy()
        processed_trade["action"] = action
        processed_trades.append(processed_trade)

    return processed_trades

def calculate_equity(session):
    try:
        response = session.get_wallet_balance(
            accountType="UNIFIED",
            coin="USDT"
        )
        usdt_data = next((coin for coin in response["result"]["list"][0]["coin"] if coin["coin"] == "USDT"), None)
        if usdt_data:
            equity = float(usdt_data["equity"])
        else:
            equity = 0.0
        return equity
    except InvalidRequestError as e:
        logging.error(f"Error calculating equity curve: {str(e)}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error calculating equity curve: {str(e)}")
        raise

def calculate_exposures_and_ratios(positions):
    long_positions = 0
    short_positions = 0
    long_exposure = 0
    short_exposure = 0

    for position in positions:
        size = float(position["size"])
        mark_price = float(position["markPrice"])
        exposure = size * mark_price

        if position["side"] == "Buy":
            long_positions += 1
            long_exposure += exposure
        else:
            short_positions += 1
            short_exposure += exposure

    return {
        "long_positions": long_positions,
        "short_positions": short_positions,
        "long_exposure": long_exposure,
        "short_exposure": short_exposure
    }

def process_fees_and_volume(session):
    today = datetime.datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    trading_fees = Decimal('0')
    funding_fees = Decimal('0')
    total_volume = Decimal('0')
    
    cursor = None

    try:
        while True:
            response = last_executions(session)
            executions = response["result"]["list"]

            for execution in executions:
                exec_time = datetime.datetime.fromtimestamp(int(execution["execTime"]) / 1000)
                if exec_time < today:
                    break

                fee = Decimal(execution["execFee"])
                if execution["execType"] == "Trade":
                    trading_fees += fee
                    total_volume += Decimal(execution["execValue"])
                elif execution["execType"] == "Funding":
                    funding_fees += fee

            cursor = response["result"].get("nextPageCursor")
            if not cursor or not executions or exec_time < today:
                break

        return {
            'trading_fees': round(trading_fees, 8),
            'funding_fees': round(funding_fees, 8),
            'total_fees': round(trading_fees + funding_fees, 8),
            'total_volume': round(total_volume, 8)
        }
    except InvalidRequestError as e:
        logging.error(f"Error processing fees and volume: {str(e)}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error processing fees and volume: {str(e)}")
        raise

def api_expiration_days(session):
    data = session.get_api_key_information()
    return data['result']['deadlineDay']

def get_account_data(account):
    """Generate account report data from various sources."""
    try:
        # Initialize API session
        session = HTTP(
            api_key=account["api_key"],
            api_secret=account["api_secret"]
        )

        # Get current date
        today = datetime.datetime.now().strftime("%Y-%m-%d")

        # Fetch all required data
        open_positions = get_all_open_positions(session)
        trades_today = get_all_trades_today(session)
        processed_trades = determine_trade_action(trades_today, open_positions)
        equity = calculate_equity(session)
        exposure_data = calculate_exposures_and_ratios(open_positions)
        fees = process_fees_and_volume(session)
        days_to_expiration = api_expiration_days(session)
        
        # Get transaction data
        deposit, withdrawal = get_existing_transaction_data(today, account["name"])

        # Construct report data
        report_data = {
            "account_name": account["name"],
            "date": today,
            "equity": equity,
            "open_positions_list": [format_position_data(pos) for pos in open_positions],
            "open_positions": len(open_positions),
            "trades_today": len(processed_trades),
            "trades_today_list": processed_trades,
            "funding_fees": fees['funding_fees'],
            "trading_fees": fees['trading_fees'],
            'total_volume': fees['total_volume'],
            "long_positions": exposure_data['long_positions'],
            "short_positions": exposure_data['short_positions'],
            "long_exposure": exposure_data['long_exposure'],
            "short_exposure": exposure_data['short_exposure'],
            "trades": processed_trades,
            "deposit": deposit,
            "withdrawal": withdrawal,
            "days_to_expiration": days_to_expiration,
        }
        
        store_daily_data(report_data)
        
        return report_data

    except Exception as e:
        logging.error(f"Error generating report for account {account['name']}: {str(e)}")
        raise

def collect_daily_data(accounts):
    result = {
        'data': {}
    }
    try:
        initialize_database()
        
        for account in accounts:
            account_name = account['name']
            logging.info(f"Collecting daily data for {account_name}...")
            data = get_account_data(account)
            result['data'][account_name] = data

            print(f"Data collected for {account_name}.")

        logging.info("Daily data collection completed successfully.")
        return result

    except Exception as e:
        logging.error(f"Error in daily data collection: {str(e)}")
        print(f"An error occurred. Please check the log file for details.")

# CALCULATIONS
def get_account_days(account_name):
    """
    Get the number of days of data available for a specific account.
    
    Args:
        account_name (str): Name of the account to query
        db_path (str): Path to the database file
        
    Returns:
        int: Number of days of data for the account
    """
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT COUNT(DISTINCT date) 
        FROM daily_reports 
        WHERE account_name = ?
        ''', (account_name,))
        
        count = cursor.fetchone()[0]
        return count
        
    except sqlite3.Error as e:
        logging.error(f"SQLite error querying days for account {account_name}: {str(e)}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error querying days for account {account_name}: {str(e)}")
        raise
    finally:
        if conn:
            try:
                conn.close()
            except Exception as e:
                logging.error(f"Error closing database connection: {str(e)}")

def process_report_data(input_data, days_lookback=7):
    """
    Process additional metrics from database using provided input data.
    
    Args:
        input_data (dict): Dictionary containing initial trading data
        days_lookback (int, optional): Number of days to look back for weekly metrics. Defaults to 7.
    
    Returns:
        dict: Additional processed data for report generation
    """
    try:
        account_name = input_data['account_name']
        report_date = input_data['date']
            
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            # Get previous week's data for comparison, finding closest date if exact match not found
            previous_date = (datetime.datetime.strptime(report_date, '%Y-%m-%d') - 
                           datetime.timedelta(days=days_lookback)).strftime('%Y-%m-%d')
            
            cursor.execute('''
                SELECT equity, date
                FROM daily_reports 
                WHERE account_name = ? 
                AND date <= ?
                ORDER BY date DESC
                LIMIT 1
            ''', (account_name, previous_date))
            
            previous_equity_row = cursor.fetchone()
            previous_equity = previous_equity_row[0] if previous_equity_row else None
            actual_previous_date = previous_equity_row[1] if previous_equity_row else None

            # Get weekly metrics
            cursor.execute('''
                SELECT 
                    COALESCE(SUM(trades_today), 0) as last_x_days_trades,
                    COALESCE(SUM(total_volume), 0) as last_x_days_volume,
                    COALESCE(SUM(funding_fees), 0) as last_x_days_funding_fees,
                    COALESCE(SUM(trading_fees), 0) as last_x_days_trading_fees
                FROM daily_reports 
                WHERE account_name = ?
                AND date BETWEEN ? AND ?
            ''', (account_name, previous_date, report_date))
            
            weekly_metrics_row = cursor.fetchone()
            weekly_metrics = {
                'last_x_days_trades': weekly_metrics_row[0] or 0,
                'last_x_days_volume': weekly_metrics_row[1] or 0,
                'last_x_days_funding_fees': weekly_metrics_row[2] or 0,
                'last_x_days_trading_fees': weekly_metrics_row[3] or 0,
            }

            # Get all time metrics
            cursor.execute('''
                SELECT 
                    MIN(date) as first_date,
                    COALESCE(SUM(deposit), 0) as total_deposits,
                    COALESCE(SUM(withdrawal), 0) as total_withdrawals,
                    COALESCE(SUM(total_volume), 0) as total_volume
                FROM daily_reports 
                WHERE account_name = ?
            ''', (account_name,))
            
            all_time_metrics = cursor.fetchone()
            first_date = datetime.datetime.strptime(all_time_metrics[0], '%Y-%m-%d')
            last_date = datetime.datetime.strptime(report_date, '%Y-%m-%d')
            days_active = (last_date - first_date).days + 1  # +1 to include both first and last day
            
        # Get all available data for equity curve, ordered by date
        cursor.execute('''
            SELECT date, equity, deposit, withdrawal
            FROM daily_reports 
            WHERE account_name = ?
            ORDER BY date
        ''', (account_name,))

        equity_curve_data = cursor.fetchall()

        # Calculate adjusted returns considering deposits and withdrawals
        adjusted_returns = []
        for i in range(len(equity_curve_data)):
            date = equity_curve_data[i][0]
            equity = equity_curve_data[i][1]
            deposit = equity_curve_data[i][2] or 0
            withdrawal = equity_curve_data[i][3] or 0
            
            if i == 0:
                daily_return = 0
            else:
                prev_equity = equity_curve_data[i-1][1]
                adjusted_equity_diff = (equity - deposit + withdrawal) - prev_equity
                daily_return = adjusted_equity_diff / prev_equity if prev_equity != 0 else 0
            
            adjusted_returns.append({
                'date': date,
                'daily_return': daily_return,
                'deposit': deposit,
                'withdrawal': withdrawal,
                'equity': equity
            })

        # Calculate total profit and adjusted return
        total_deposits = all_time_metrics[1]
        total_withdrawals = all_time_metrics[2]
        net_investment = total_deposits - total_withdrawals
        total_profit = input_data['equity'] - net_investment
        
        # Calculate total adjusted return (similar to the previous calculation)
        cumulative_return = 1
        for return_data in adjusted_returns[1:]:  # Skip first day as it has 0 return
            cumulative_return *= (1 + return_data['daily_return'])
        total_adjusted_return = cumulative_return - 1
        annualized_return = (1 + total_adjusted_return) ** (365 / days_active) - 1

        # Calculate additional metrics not in input data
        additional_data = {
            'long_ratio': (input_data['long_positions'] / 
                        (input_data['long_positions'] + input_data['short_positions']) * 100
                        if input_data['long_positions'] + input_data['short_positions'] > 0 else 0),
            'short_ratio': (input_data['short_positions'] / 
                         (input_data['long_positions'] + input_data['short_positions']) * 100
                         if input_data['long_positions'] + input_data['short_positions'] > 0 else 0),
            'overall_exposure': input_data['long_exposure'] - input_data['short_exposure'],
            'previous_week_equity_usdt': previous_equity,
            'previous_week_date': actual_previous_date,
            'equity_difference_usdt': ((input_data['equity'] - previous_equity) / previous_equity * 100
                                    if previous_equity else None),
            'adjusted_returns': adjusted_returns,
            **weekly_metrics,
            'last_x_days_total_fees': ((weekly_metrics['last_x_days_funding_fees'] or 0) + 
                                    (weekly_metrics['last_x_days_trading_fees'] or 0)),
            # New metrics
            'days_active': days_active,
            'total_deposit': total_deposits,
            'total_withdraw': total_withdrawals,
            'total_profit': total_profit,
            'total_volume': all_time_metrics[3],
            'total_adjusted_return': total_adjusted_return,
            'annualized_return': annualized_return
        }
        
        return additional_data

    except sqlite3.Error as e:
        logging.error(f"Database error while preparing report data: {str(e)}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error while preparing report data: {str(e)}")
        raise

# ACCOUNT REPORT
def create_table(data, headers=None, col_widths=None, primary_color=HexColor("#2a5e35"), 
                secondary_color=HexColor("#E2E2E2")):
    """Generic table creation function"""
    if headers:
        data.insert(0, headers)
    
    if col_widths:
        table = Table(data, colWidths=col_widths)
    else:
        table = Table(data)

    style = [
        ('BACKGROUND', (0, 0), (-1, 0), primary_color),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('TOPPADDING', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), secondary_color),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('TOPPADDING', (0, 1), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]
    
    table.setStyle(TableStyle(style))
    return table

def create_account_summary(report_data):
    """Create account summary table"""
    data = [
        ["Current Equity", f"{report_data['equity']:,.2f} USDT"],
        ["Open Positions", report_data['open_positions']],
        ["Trades This Week", report_data['last_x_days_trades']],
        ["Long Positions", f"{report_data['long_positions']} ({report_data['long_ratio']:.2f}%)"],
        ["Short Positions", f"{report_data['short_positions']} ({report_data['short_ratio']:.2f}%)"],
        ["Long Exposure", f"{report_data['long_exposure']:,.2f} USDT"],
        ["Short Exposure", f"{report_data['short_exposure']:,.2f} USDT"],
        ["Net Exposure", f"{report_data['long_exposure'] - report_data['short_exposure']:,.2f} USDT"],
        ["Keys Expiration", f"{report_data['days_to_expiration']} days"],
    ]
    headers = ["Metric", "Value"]
    return create_table(data, headers=headers, col_widths=[3*inch, 2*inch])

def create_performance_metrics(report_data):
    """Create account summary table"""
    data = [
        ["N. Trades", f"{report_data['last_x_days_trades']}"],
        ["Volume", f"{report_data['last_x_days_volume']:,.2f}"],
        ["Funding Fees", f"{report_data['last_x_days_funding_fees']:,.2f}"],
        ["Trading Fees", f"{report_data['last_x_days_trading_fees']:,.2f}"],
        ["Total Fees", f"{report_data['last_x_days_total_fees']:,.2f}"],
        ["Previous Week Equity", f"{report_data['previous_week_equity_usdt']:,.2f} USDT" if report_data['previous_week_equity_usdt'] is not None else "N/A"],
        ["Performance This Week", f"{report_data['equity_difference_usdt']:.2f} %" if report_data['equity_difference_usdt'] is not None else "N/A"],
    ]
    headers = ["Metric (This Week)", "Value"]
    return create_table(data, headers=headers, col_widths=[3*inch, 2*inch])

def create_overall_performance_metrics(report_data):
    """Create account summary table"""
    data = [
        ["Days Active", f"{report_data['days_active']}"],
        ["Total Invested", f"{report_data['total_deposit']:,.2f}"],
        ["Total Withdrawn", f"{report_data['total_withdraw']:,.2f}"],
        ["Total Profit", f"{report_data['total_profit']:,.2f}"],
        ["Total Volume", f"{report_data['total_volume']:,.2f}"],
        ["Adjusted Return", f"{report_data['total_adjusted_return']*100:,.2f}%"],
        ["Annualized Return", f"{report_data['annualized_return']*100:,.2f}%"],
    ]
    headers = ["Metric", "Value"]
    return create_table(data, headers=headers, col_widths=[3*inch, 2*inch])

def create_open_positions_table(positions, styles):
    """Create open positions table"""
    if not positions:
        return Paragraph("\nNo open positions.\n", styles['Normal'])
        
    data = []
    for position in positions:
        data.append([
            position["symbol"],
            ("Long" if position["side"] == "Buy" else "Short"),
            f"{position['exposure']:,.2f} USDT",
            f"{position['entry_price']:,.4f}",
            f"{position['mark_price']:,.4f}",
            f"{position['unrealized_pnl']:,.2f} USDT",
            position["leverage"]
        ])
        
    headers = ["Symbol", "Side", "Exposure", "Entry Price", "Mark Price", "Unrealized PNL", "Leverage"]
    return create_table(data, headers=headers)

def create_performance_graph(adjusted_returns, primary_color, styles, days_to_show=30):
   """
   Create performance graph showing equity curve adjusted for deposits/withdrawals
   
   Args:
       adjusted_returns (list): List of daily return data
       primary_color: Color for the graph line
       styles: Report styles
       days_to_show (int): Number of days of data to display. Defaults to 30.
   """
   if not adjusted_returns or len(adjusted_returns) < 2:
       return Paragraph("\nCannot generate graph.\n", styles['Normal'])
   
   # Filter for the requested number of days
   end_date = datetime.datetime.strptime(adjusted_returns[-1]['date'], '%Y-%m-%d')
   start_date = end_date - datetime.timedelta(days=days_to_show)
   
   filtered_returns = [
       data for data in adjusted_returns 
       if datetime.datetime.strptime(data['date'], '%Y-%m-%d') >= start_date
   ]
   
   if not filtered_returns:
       return Paragraph("\nNo data available for selected period.\n", styles['Normal'])
       
   plt.figure(figsize=(10, 5))
   dates = [datetime.datetime.strptime(data['date'], '%Y-%m-%d') 
           for data in filtered_returns]
   
   # Calculate normalized values using pre-calculated daily returns
   normalized_values = [100]  # Start at 100
   for data in filtered_returns[1:]:  # Skip first entry since its return is 0
       normalized_values.append(normalized_values[-1] * (1 + data['daily_return']))
   
   # Convert HexColor to matplotlib format
   primary_color_rgb = primary_color.rgb()
   matplotlib_color = (primary_color_rgb[0]/255, primary_color_rgb[1]/255, primary_color_rgb[2]/255)
   
   plt.plot(dates, normalized_values, marker='o', linestyle='-', color=matplotlib_color)
   plt.title(f'Portfolio Performance ({days_to_show} Days)')
   plt.xlabel('Date')
   plt.ylabel('Value (Starting at 100)')
   plt.grid(True, alpha=0.3)
   plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
   plt.gcf().autofmt_xdate()
   plt.axhline(y=100, color='r', linestyle='--', alpha=0.5)
   
   # Add y-axis formatting
   plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}'))
   
   img_buffer = BytesIO()
   plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
   img_buffer.seek(0)
   plt.close()

   return Image(img_buffer, width=8*inch, height=4*inch)

def create_equity_graph(adjusted_returns, styles, days_to_show=None):
   """
   Create graph showing equity and cumulative deposits/withdrawals over time
   
   Args:
       adjusted_returns (list): List of daily return data
       primary_color: Color for the graph line
       styles: Report styles
       days_to_show (int, optional): Number of days of data to display. 
                                   If None, shows complete history. Defaults to None.
   """
   if not adjusted_returns or len(adjusted_returns) < 2:
       return Paragraph("\nCannot generate graph.\n", styles['Normal'])
   
   # Filter data if days_to_show is specified
   if days_to_show is not None:
       end_date = datetime.datetime.strptime(adjusted_returns[-1]['date'], '%Y-%m-%d')
       start_date = end_date - datetime.timedelta(days=days_to_show)
       
       filtered_returns = [
           data for data in adjusted_returns 
           if datetime.datetime.strptime(data['date'], '%Y-%m-%d') >= start_date
       ]
   else:
       filtered_returns = adjusted_returns  # Use all data
   
   if not filtered_returns:
       return Paragraph("\nNo data available for selected period.\n", styles['Normal'])
       
   plt.figure(figsize=(10, 5))
   dates = [datetime.datetime.strptime(data['date'], '%Y-%m-%d') 
           for data in filtered_returns]
   
   # Get equity values
   equity_values = [data['equity'] for data in filtered_returns]
   
   # Calculate cumulative deposits/withdrawals
   cumulative_flow = []
   current_total = 0

   for data in filtered_returns:
       net_flow = data.get('deposit', 0) - data.get('withdrawal', 0)  # Using get() with default 0
       current_total += net_flow
       cumulative_flow.append(current_total)
   
   # Create figure with two lines
   plt.plot(dates, equity_values, linestyle='-', color='#2a5e35', 
           label='Account Equity')
   plt.plot(dates, cumulative_flow, linestyle='--', color='gray', 
           label='Cumulative Deposits/Withdrawals')
   
   # Set title based on whether showing complete history or specific period
   title = 'Account Equity vs Deposits/Withdrawals'
   if days_to_show:
       title += f' ({days_to_show} Days)'
   else:
       title += ' (Complete History)'
   
   plt.title(title)
   plt.xlabel('Date')
   plt.ylabel('Value (USDT)')
   plt.grid(True, alpha=0.3)
   plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
   plt.gcf().autofmt_xdate()
   
   # Add legend
   plt.legend(loc='best')
   
   # Format y-axis to show thousands with K
   plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1000:.1f}K'))
   
   img_buffer = BytesIO()
   plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
   img_buffer.seek(0)
   plt.close()

   return Image(img_buffer, width=8*inch, height=4*inch)

def generate_weekly_report(all_data):
    """Generate a daily trading report PDF"""
    try:
        # Default configuration
        config = {
            'primary_color': HexColor("#2a5e35"),
            'secondary_color': HexColor("#E2E2E2"),
            'page_size': landscape(letter),
            'margins': {'right': 72, 'left': 72, 'top': 72, 'bottom': 18}
        }

        # Create directories
        reports_dir = 'reports'
        account_dir = os.path.join(reports_dir, all_data['account_name'])
        os.makedirs(account_dir, exist_ok=True)
        
        # Generate filepath
        filename = f"{all_data['account_name']}_{all_data['date']}.pdf"
        filepath = os.path.join(account_dir, filename)
        
        # Initialize document
        doc = SimpleDocTemplate(
            filepath, 
            pagesize=config['page_size'],
            rightMargin=config['margins']['right'],
            leftMargin=config['margins']['left'],
            topMargin=config['margins']['top'],
            bottomMargin=config['margins']['bottom']
        )
        
         # Build elements list
        elements = []

        # Initialize styles with unique names
        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle(name='Center', alignment=1, textColor=config['primary_color'], fontSize=15))

        # Title
        title_style = ParagraphStyle(
            'Title',
            parent=styles['Title'],
            fontSize=24,
            textColor=config['primary_color'],
            spaceAfter=12
        )
        elements.extend([
            Paragraph(f"Daily Trading Report - {all_data['date']}", title_style),
            Paragraph(f"{all_data['account_name'].upper()}", styles['Center']),
            Spacer(1, 0.25*inch)
        ])
        
        # Add sections
        sections = [
            ("1. Account Summary", create_account_summary(all_data)),
            ("2. Open Positions", create_open_positions_table(all_data['open_positions_list'], styles)),
            ("3. Weekly Performance Metrics", create_performance_metrics(all_data)),
            ("4. Equity Curve", create_equity_graph(all_data.get('adjusted_returns'), styles)),
            ("5. Performance Analysis", create_performance_graph(all_data.get('adjusted_returns'), config['primary_color'], styles)),
            ("6. Overall Performance", create_overall_performance_metrics(all_data)),
        ]
        
        for title, content in sections:
            elements.extend([
                Paragraph(title, styles['Heading2']),
                Spacer(1, 0.25*inch),
                content,
                Spacer(1, 0.25*inch),
                PageBreak(),
            ])
        
        # Build document
        doc.build(elements)
        logging.info(f"Report generated successfully: {filepath}")
        return filepath
        
    except Exception as e:
        logging.error(f"Error generating report: {str(e)}")
        raise

# COMBINED REPORT
def create_combined_summary(accounts_data):
    """
    Create a comparative account summary table for multiple accounts with metrics as columns
    
    Parameters:
    accounts_data (list): List of dictionaries containing account data
    
    Returns:
    Table: A formatted table comparing metrics across accounts
    """
    # Define the metrics we want to show and their formatting
    metric_formats = {
        "Equity": lambda x: f"{x['equity']:,.2f}",
        "Positions": lambda x: str(x['open_positions']),
        "Trades This Week": lambda x: str(x['last_x_days_trades']),
        "Long Positions": lambda x: f"{x['long_positions']} ({x['long_ratio']:.2f}%)",
        "Short Positions": lambda x: f"{x['short_positions']} ({x['short_ratio']:.2f}%)",
        "Long Exposure": lambda x: f"{x['long_exposure']:,.2f}",
        "Short Exposure": lambda x: f"{x['short_exposure']:,.2f}",
        "Net Exposure": lambda x: f"{x['long_exposure'] - x['short_exposure']:,.2f}",
        "API": lambda x: x['days_to_expiration'],
    }
    
    # Create headers with metrics
    headers = ["Account"] + list(metric_formats.keys())
    
    # Prepare data rows (one row per account)
    data = []
    for account in accounts_data:
        row = [account["account_name"]]  # First column is the account name
        # Add value for each metric
        for metric, format_func in metric_formats.items():
            try:
                formatted_value = format_func(account)
                row.append(formatted_value)
            except KeyError:
                row.append("N/A")  # Handle missing data gracefully
        data.append(row)
    
    return create_table(data, headers=headers)

def create_combined_performance_metrics(accounts_data):
    """
    Create a comparative account performance metrics table with metrics as columns
    
    Parameters:
    accounts_data (list): List of dictionaries containing account data
    
    Returns:
    Table: A formatted table comparing metrics across accounts
    """
    # Define the metrics we want to show and their formatting
    metric_formats = {
        "Equity": lambda x: f"{x['equity']:,.2f}",
        "N. Trades": lambda x: f"{x['last_x_days_trades']}",
        "Volume": lambda x: f"{x['last_x_days_volume']:,.2f}",
        "Funding Fees": lambda x: f"{x['last_x_days_funding_fees']:,.2f}",
        "Trading Fees": lambda x: f"{x['last_x_days_trading_fees']:,.2f}",
        "Total Fees": lambda x: f"{x['last_x_days_funding_fees']:,.2f}",
        "Previous Week Equity": lambda x: f"{x['previous_week_equity_usdt']:,.2f} USDT" if x['previous_week_equity_usdt'] is not None else "N/A",
        "Return": lambda x: f"{x['equity_difference_usdt']:,.2f}%" if x['equity_difference_usdt'] is not None else "N/A",
    }
    
    # Create headers with metrics
    headers = ["Account"] + list(metric_formats.keys())
    
    # Prepare data rows (one row per account)
    data = []
    for account in accounts_data:
        row = [account["account_name"]]  # First column is the account name
        # Add value for each metric
        for metric, format_func in metric_formats.items():
            try:
                formatted_value = format_func(account)
                row.append(formatted_value)
            except KeyError:
                row.append("N/A")  # Handle missing data gracefully
        data.append(row)
    
    return create_table(data, headers=headers)

def create_combined_overall_perf_metrics(accounts_data):
    """
    Create a comparative account summary table for multiple accounts with metrics as columns
    
    Parameters:
    accounts_data (list): List of dictionaries containing account data
    
    Returns:
    Table: A formatted table comparing metrics across accounts
    """
    # Define the metrics we want to show and their formatting
    metric_formats = {
        "Equity": lambda x: f"{x['equity']:,.2f}",
        "Days Active": lambda x: f"{x['days_active']}",
        "Total Invested": lambda x: f"{x['total_deposit']:,.2f}",
        "Total Withdrawn": lambda x: f"{x['total_withdraw']:,.2f}",
        "Total Profit": lambda x: f"{x['total_profit']:,.2f}",
        "Total Volume": lambda x: f"{x['total_volume']:,.2f}",
        "Adjusted Return": lambda x: f"{x['total_adjusted_return']*100:,.2f}%",
        "Annualized Return": lambda x: f"{x['annualized_return']*100:,.2f}%",
    }
    
    # Create headers with metrics
    headers = ["Account"] + list(metric_formats.keys())
    
    # Prepare data rows (one row per account)
    data = []
    for account in accounts_data:
        row = [account["account_name"]]  # First column is the account name
        # Add value for each metric
        for metric, format_func in metric_formats.items():
            try:
                formatted_value = format_func(account)
                row.append(formatted_value)
            except KeyError:
                row.append("N/A")  # Handle missing data gracefully
        data.append(row)
    
    return create_table(data, headers=headers)

def create_combined_perf_graph(accounts_data_list, primary_color, styles, days_to_show=30):
    """
    Create performance graph showing equity curves adjusted for deposits/withdrawals for multiple accounts
    
    Args:
        accounts_data_list (list): List of dictionaries containing account data
        primary_color: Base color for the graph lines
        styles: Report styles
        days_to_show (int): Number of days of data to display. Defaults to 30.
    """
    if not accounts_data_list or len(accounts_data_list) == 0:
        return Paragraph("\nNo account data available.\n", styles['Normal'])
        
    # Set up the plot
    plt.figure(figsize=(12, 6))
    
    # Generate distinct colors for each account
    # Create color variations based on primary color
    base_color_rgb = primary_color.rgb()
    base_color_hsv = colorsys.rgb_to_hsv(base_color_rgb[0]/255, base_color_rgb[1]/255, base_color_rgb[2]/255)
    num_accounts = len(accounts_data_list)
    colors = []
    
    for i in range(num_accounts):
        # Adjust hue while keeping saturation and value similar
        hue = (base_color_hsv[0] + (i * 0.7/num_accounts)) % 1.0
        rgb = colorsys.hsv_to_rgb(hue, base_color_hsv[1], base_color_hsv[2])
        colors.append(rgb)
    
    # Find the latest end date across all accounts
    latest_end_date = max(
        datetime.datetime.strptime(account['adjusted_returns'][-1]['date'], '%Y-%m-%d')
        for account in accounts_data_list
    )
    start_date = latest_end_date - datetime.timedelta(days=days_to_show)
    
    # Plot each account's performance
    for idx, account_data in enumerate(accounts_data_list):
        adjusted_returns = account_data['adjusted_returns']
        account_name = account_data.get('account_name', f'Account {idx + 1}')
        
        # Filter data for the requested time period
        filtered_returns = [
            data for data in adjusted_returns 
            if datetime.datetime.strptime(data['date'], '%Y-%m-%d') >= start_date
        ]
        
        if not filtered_returns:
            continue
            
        dates = [datetime.datetime.strptime(data['date'], '%Y-%m-%d') 
                for data in filtered_returns]
        
        # Calculate normalized values
        normalized_values = [100]  # Start at 100
        for data in filtered_returns[1:]:
            normalized_values.append(normalized_values[-1] * (1 + data['daily_return']))
        
        # Plot the line for this account
        plt.plot(dates, normalized_values, 
                # marker='o', 
                # markersize=4,
                linestyle='-', 
                label=account_name,
                linewidth=2)
    
    # Customize the plot
    plt.title(f'Portfolio Performance Comparison ({days_to_show} Days)')
    plt.xlabel('Date')
    plt.ylabel('Value (Starting at 100)')
    plt.grid(True, alpha=0.3)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gcf().autofmt_xdate()
    plt.axhline(y=100, color='gray', linestyle='--', alpha=0.5)
    
    # Add legend
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Format y-axis
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}'))
    
    # Adjust layout to prevent legend cutoff
    plt.tight_layout()
    
    # Save to buffer
    img_buffer = BytesIO()
    plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
    img_buffer.seek(0)
    plt.close()

    return Image(img_buffer, width=10*inch, height=5*inch)  # Slightly larger to accommodate legend

def generate_combined_report(data):
    accounts_data_list = list(data.values())
    try:
        # Default configuration
        config = {
            'primary_color': HexColor("#2a5e35"),
            'secondary_color': HexColor("#E2E2E2"),
            'page_size': landscape(letter),
            'margins': {'right': 72, 'left': 72, 'top': 72, 'bottom': 18}
        }

        # Create directories
        reports_dir = 'reports'
        account_dir = os.path.join(reports_dir, 'total')
        os.makedirs(account_dir, exist_ok=True)
        
        # Generate filepath
        filename = f"combined_report_{accounts_data_list[0]['date']}.pdf"
        filepath = os.path.join(account_dir, filename)
        
        # Initialize document
        doc = SimpleDocTemplate(
            filepath, 
            pagesize=config['page_size'],
            rightMargin=config['margins']['right'],
            leftMargin=config['margins']['left'],
            topMargin=config['margins']['top'],
            bottomMargin=config['margins']['bottom']
        )
        
        # Build elements list
        elements = []

        # Initialize styles with unique names
        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle(name='Center', alignment=1, textColor=config['primary_color'], fontSize=15))

        # Title
        title_style = ParagraphStyle(
            'Title',
            parent=styles['Title'],
            fontSize=24,
            textColor=config['primary_color'],
            spaceAfter=12
        )
        elements.extend([
            Paragraph(f"Daily Combined Report - {accounts_data_list[0]['date']}", title_style),
            Spacer(1, 0.25*inch)
        ])
        
        # Add sections
        sections = [
            ("1. Combined Summary", create_combined_summary(accounts_data_list)),
            ("2. Weekly Performance Metrics", create_combined_performance_metrics(accounts_data_list)),
            # ("4. Equity Curve", create_equity_graph(data.get('adjusted_returns'), styles)),
            ("3. Overall Performance", create_combined_overall_perf_metrics(accounts_data_list)),
            ("4. Performance Analysis", create_combined_perf_graph(accounts_data_list, config['primary_color'], styles)),
        ]
        
        for title, content in sections:
            elements.extend([
                Paragraph(title, styles['Heading2']),
                Spacer(1, 0.25*inch),
                content,
                Spacer(1, 0.25*inch),
                PageBreak(),
            ])
        
        # Build document
        doc.build(elements)
        logging.info(f"Report generated successfully: {filepath}")
        return filepath
        
    except Exception as e:
        logging.error(f"Error generating report: {str(e)}")
        raise

# PROCESS REPORT
def weekly_report(data):
    accounts_data_list = list(data['data'].values())
    all_data_combined = {}
    for account_data in accounts_data_list:
        logging.info(f"Generating weekly report for {account_data['account_name']}...")
        try:
            processed_data = process_report_data(account_data)
            all_data = account_data.copy()
            all_data.update(processed_data)
            all_data_combined[account_data['account_name']] = all_data
            filepath = generate_weekly_report(all_data)
            print(f"Report generated: {filepath}")
        except Exception as e:
            print(f"Failed to generate report: {e}")
    return all_data_combined

def weekly_report_all(all_data):
    all_account_names = list(all_data.keys())
    # print(all_account_names)
    logging.info(f"Generating weekly combined report for {len(all_account_names)} accounts...")
    try:
        filepath = generate_combined_report(all_data)
        print(f"Report generated: {filepath}")
    except Exception as e:
        print(f"Failed to generate report: {e}")

# RUN
def main2():
    # script to use with cronjob
    parser = argparse.ArgumentParser(description='Trading Report Generator')
    parser.add_argument('--force-weekly', action='store_true', help='Force generate weekly report')
    args = parser.parse_args()

    accounts = get_accounts_from_env()
    data = collect_daily_data(accounts)

    # Check if it's Sunday (weekday() returns 6 for Sunday) or if --force-weekly flag is used
    # to change the day bare in mind: 0=monday, 6=sunday, ecc
    if datetime.datetime.now().weekday() == 6 or args.force_weekly:
        all_data = weekly_report(data)
        weekly_report_all(all_data)

def main():
    accounts = get_accounts_from_env()
    print(f"Active accounts: {len(accounts)}")
    data = collect_daily_data(accounts)

    all_data = weekly_report(data)
    weekly_report_all(all_data)

if __name__ == "__main__":
    main()