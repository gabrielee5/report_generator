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
from reportlab.platypus import Image
import sqlite3
import os
import csv
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from io import BytesIO
from decimal import Decimal
import logging
import argparse
import statistics
from combined_report import generate_combined_report, daily_combined_report

# TO DO
# make the graphic of the report better

os.chdir(os.path.dirname(os.path.abspath(__file__)))

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
                    "api_secret": env_vars[f"{account_id}_api_secret"],
                    "email": env_vars.get(f"{account_id}_email", "")  # Add email, default to empty string if not found
                })
        
        if not accounts:
            raise ValueError("No accounts found in .env file")
        
        return accounts
    except Exception as e:
        logging.error(f"Error reading accounts from .env: {str(e)}")
        raise

def get_account_type(session):
    try:
        # Try to access unified account endpoint
        session.get_wallet_balance(accountType="UNIFIED")
        return "Unified"
    except:
        # If it fails, assume it's a standard account
        return "Standard"
    
# DATABASE
def initialize_database():
    try:
        conn = sqlite3.connect('database.db')
        cursor = conn.cursor()

        cursor.execute('''
        CREATE TABLE IF NOT EXISTS daily_reports (
            date TEXT,
            account_name TEXT,
            equity REAL,
            equity_btc REAL,
            open_positions INTEGER,
            trades_today INTEGER,
            long_positions INTEGER,
            short_positions INTEGER,
            long_exposure REAL,
            short_exposure REAL,
            net_exposure REAL,
            funding_fees REAL,
            trading_fees REAL,
            total_fees REAL,
            total_volume REAL,
            deposit REAL DEFAULT 0,
            withdrawal REAL DEFAULT 0,
            PRIMARY KEY (date, account_name)
        )
        ''')

        # Add new columns if they don't exist
        cursor.execute("PRAGMA table_info(daily_reports)")
        columns = [column[1] for column in cursor.fetchall()]
        
        if 'deposit' not in columns:
            cursor.execute("ALTER TABLE daily_reports ADD COLUMN deposit REAL DEFAULT 0")
        if 'withdrawal' not in columns:
            cursor.execute("ALTER TABLE daily_reports ADD COLUMN withdrawal REAL DEFAULT 0")

        conn.commit()
    except sqlite3.Error as e:
        logging.error(f"Database error: {str(e)}")
        raise
    finally:
        if conn:
            conn.close()

def store_daily_report(report_data):
    try:
        conn = sqlite3.connect('database.db')
        cursor = conn.cursor()
        
        for key, value in report_data.items():
            if isinstance(value, Decimal):
                report_data[key] = float(value)

        # Check for existing record
        cursor.execute('''
        SELECT deposit, withdrawal FROM daily_reports
        WHERE date = ? AND account_name = ?
        ''', (report_data['date'], report_data['account_name']))
        existing_data = cursor.fetchone()

        if existing_data:
            # Update existing record, preserving deposit and withdrawal
            cursor.execute('''
            UPDATE daily_reports 
            SET equity = ?, equity_btc = ?, open_positions = ?, trades_today = ?, 
                long_positions = ?, short_positions = ?, long_exposure = ?, 
                short_exposure = ?, net_exposure = ?, funding_fees = ?, 
                trading_fees = ?, total_fees = ?, total_volume = ?
            WHERE date = ? AND account_name = ?
            ''', (
                report_data['equity'],
                report_data['equity_btc'],
                len(report_data['open_positions']),
                report_data['trades_today'],
                report_data['long_positions'],
                report_data['short_positions'],
                report_data['long_exposure'],
                report_data['short_exposure'],
                report_data['overall_exposure'],
                report_data['funding_fees'],
                report_data['trading_fees'],
                report_data['total_fees'],
                report_data['total_volume'],
                report_data['date'],
                report_data['account_name']
            ))
        else:
            # Insert new record
            cursor.execute('''
            INSERT INTO daily_reports 
            (date, account_name, equity, equity_btc, open_positions, trades_today, 
            long_positions, short_positions, long_exposure, short_exposure, 
            net_exposure, funding_fees, trading_fees, total_fees, total_volume, 
            deposit, withdrawal)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                report_data['date'],
                report_data['account_name'],
                report_data['equity'],
                report_data['equity_btc'],
                len(report_data['open_positions']),
                report_data['trades_today'],
                report_data['long_positions'],
                report_data['short_positions'],
                report_data['long_exposure'],
                report_data['short_exposure'],
                report_data['overall_exposure'],
                report_data['funding_fees'],
                report_data['trading_fees'],
                report_data['total_fees'],
                report_data['total_volume'],
                report_data.get('deposit', 0),
                report_data.get('withdrawal', 0)
            ))
        
        conn.commit()
    except sqlite3.Error as e:
        logging.error(f"Database error: {str(e)}")
        raise
    finally:
        if conn:
            conn.close()

# GET AND MANAGE DATA
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
    account_type = get_account_type(session)
    try:
        if account_type == "Unified":
            response = session.get_wallet_balance(
                accountType="UNIFIED",
                coin="USDT"
            )
        else:
            response = session.get_wallet_balance(
                accountType="CONTRACT",
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

    total_positions = long_positions + short_positions
    overall_exposure = long_exposure - short_exposure

    long_ratio = (long_positions / total_positions) * 100 if total_positions > 0 else 0
    short_ratio = 100 - long_ratio

    return {
        "long_positions": long_positions,
        "short_positions": short_positions,
        "long_exposure": long_exposure,
        "short_exposure": short_exposure,
        "long_ratio": long_ratio,
        "short_ratio": short_ratio,
        "overall_exposure": overall_exposure
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

def get_historical_equity(account_name, days=30):
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()

    query = '''
    SELECT date, equity
    FROM daily_reports
    WHERE account_name = ?
    ORDER BY date DESC
    LIMIT ?
    '''
    
    cursor.execute(query, (account_name, days))
    rows = cursor.fetchall()
    conn.close()

    # Convert to list of tuples (datetime, equity)
    data = [(datetime.datetime.strptime(row[0], '%Y-%m-%d'), row[1]) for row in rows]
    return sorted(data)  # Sort by date

def create_equity_curve_plot(account_name, days=30):
    data = get_historical_equity(account_name, days)
    dates, equity = zip(*data)

    plt.figure(figsize=(10, 5))
    plt.plot(dates, equity, marker='o', color='#2a5e35') # same color as report
    plt.title(f'Equity Curve - {account_name}')
    plt.xlabel('Date')
    plt.ylabel('Equity (USDT)')
    plt.grid(True)

    # Format x-axis to show dates nicely
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gcf().autofmt_xdate()  # Rotate and align the tick labels

    # Save plot to a BytesIO object
    img_buffer = BytesIO()
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    plt.close()

    return img_buffer

def equity_btc(session, equity_usdt):
    response = session.get_tickers(
    category="spot",
    symbol="BTCUSDT",
    )
    last_price_btc = float(response["result"]["list"][0]["lastPrice"])
    return float(equity_usdt) / float(last_price_btc)

def get_previous_week_equity(account_name):
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()

    one_week_ago = (datetime.datetime.now() - datetime.timedelta(days=7)).strftime('%Y-%m-%d')

    query = '''
    SELECT equity, equity_btc
    FROM daily_reports
    WHERE account_name = ? AND date = ?
    '''
    
    cursor.execute(query, (account_name, one_week_ago))
    result = cursor.fetchone()
    conn.close()

    if result:
        return {"equity_usdt": result[0], "equity_btc": result[1]}
    else:
        return None

def get_last_x_days_metrics(account_name, x_days=7):
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()

    x_days_ago = (datetime.datetime.now() - datetime.timedelta(days=x_days)).strftime('%Y-%m-%d')

    query = '''
    SELECT SUM(funding_fees), SUM(trading_fees), SUM(total_fees), SUM(trades_today), SUM(total_volume)
    FROM daily_reports
    WHERE account_name = ? AND date >= ?
    '''
    
    cursor.execute(query, (account_name, x_days_ago))
    result = cursor.fetchone()
    conn.close()

    funding_fees = result[0] if result[0] is not None else 0
    trading_fees = result[1] if result[1] is not None else 0
    fees = result[2] if result[2] is not None else 0
    trades = result[3] if result[3] is not None else 0
    volume = result[4] if result[4] is not None else 0

    return {
        'funding_fees': funding_fees,
        'trading_fees': trading_fees,
        'total_fees': fees,
        'trades': trades,
        'volume': volume
    }

def parse_date(date_string):
    from datetime import datetime
    return datetime.strptime(date_string, '%Y-%m-%d %H:%M:%S')

def process_transactions_csv():
    conn = None
    try:
        conn = sqlite3.connect('database.db')
        cursor = conn.cursor()
        
        with open('transactions.csv', 'r') as csvfile:
            csvreader = csv.DictReader(csvfile)
            for row in csvreader:
                timestamp = parse_date(row['Timestamp'])
                date = timestamp.strftime('%Y-%m-%d')
                account_name = row['Account Name']
                amount = float(row['Amount'])
                transaction_type = row['Type'].lower()

                logging.info(f"Processing transaction: {date}, {account_name}, {amount}, {transaction_type}")

                # Check if there's an existing record for this date, account, and transaction type
                cursor.execute('''
                SELECT deposit, withdrawal FROM daily_reports
                WHERE date = ? AND account_name = ?
                ''', (date, account_name))
                existing_record = cursor.fetchone()

                if existing_record:
                    existing_deposit, existing_withdrawal = existing_record
                    if transaction_type == 'deposit' and existing_deposit != amount:
                        cursor.execute('''
                        UPDATE daily_reports
                        SET deposit = ?
                        WHERE date = ? AND account_name = ?
                        ''', (amount, date, account_name))
                    elif transaction_type == 'withdrawal' and existing_withdrawal != amount:
                        cursor.execute('''
                        UPDATE daily_reports
                        SET withdrawal = ?
                        WHERE date = ? AND account_name = ?
                        ''', (amount, date, account_name))
                    else:
                        logging.info(f"Transaction already exists: {date}, {account_name}, {amount}, {transaction_type}")
                        continue
                else:
                    # Insert new record
                    if transaction_type == 'deposit':
                        cursor.execute('''
                        INSERT INTO daily_reports (date, account_name, deposit, withdrawal)
                        VALUES (?, ?, ?, 0)
                        ''', (date, account_name, amount))
                    elif transaction_type == 'withdrawal':
                        cursor.execute('''
                        INSERT INTO daily_reports (date, account_name, deposit, withdrawal)
                        VALUES (?, ?, 0, ?)
                        ''', (date, account_name, amount))

                # Log the number of rows affected
                rows_affected = cursor.rowcount
                logging.info(f"Rows affected: {rows_affected}")

                # Verify the data was written
                cursor.execute('''
                SELECT deposit, withdrawal FROM daily_reports
                WHERE date = ? AND account_name = ?
                ''', (date, account_name))
                result = cursor.fetchone()
                if result:
                    logging.info(f"After operation - Deposit: {result[0]}, Withdrawal: {result[1]}")
                else:
                    logging.warning(f"No data found after operation for {date}, {account_name}")

        conn.commit()
        logging.info("All transactions processed and committed to database.")
        print("Transactions processed successfully.")
    except Exception as e:
        if conn:
            conn.rollback()
        logging.error(f"Error processing transactions CSV: {str(e)}")
        print(f"An error occurred while processing transactions. Please check the log file for details.")
    finally:
        if conn:
            conn.close()

def calculate_adjusted_returns(account_name, start_date=None, end_date=None):
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()

    # Construct the date range condition
    date_condition = ""
    params = [account_name]
    if start_date:
        date_condition += " AND date >= ?"
        params.append(start_date)
    if end_date:
        date_condition += " AND date <= ?"
        params.append(end_date)

    # Fetch the data from the database
    query = f'''
    SELECT date, equity, deposit, withdrawal
    FROM daily_reports
    WHERE account_name = ?{date_condition}
    ORDER BY date
    '''
    
    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        logging.warning(f"No data found for account {account_name} in the specified date range.")
        return []

    if len(rows) < 2:
        logging.warning(f"Insufficient data for account {account_name} in the date range. At least two data points are required.")
        return []

    # Process the data
    adjusted_equity = []
    cumulative_deposit = 0
    cumulative_withdrawal = 0
    previous_adjusted_equity = None

    for date, equity, deposit, withdrawal in rows:
        # Use 0 as default value if deposit, withdrawal, or equity is None
        equity = equity or 0
        deposit = deposit or 0
        withdrawal = withdrawal or 0
        
        cumulative_deposit += deposit
        cumulative_withdrawal += withdrawal
        adjusted_equity_value = equity - (cumulative_deposit - cumulative_withdrawal)

        if previous_adjusted_equity is not None:
            daily_return = (adjusted_equity_value / previous_adjusted_equity) - 1
        else:
            daily_return = 0

        adjusted_equity.append({
            'date': date,
            'equity': equity,
            'adjusted_equity': adjusted_equity_value,
            'daily_return': daily_return
        })

        previous_adjusted_equity = adjusted_equity_value

    logging.info(f"Processed {len(adjusted_equity)} data points for account {account_name}")
    return adjusted_equity

# REPORT
def export_report_to_pdf(report_data):
    # Create 'reports' directory if it doesn't exist
    reports_dir = 'reports'
    os.makedirs(reports_dir, exist_ok=True)

    # Create account-specific directory
    account_dir = os.path.join(reports_dir, report_data['account_name'])
    os.makedirs(account_dir, exist_ok=True)

    # Generate filename with current date
    filename = f"{report_data['account_name']}_{report_data['date']}.pdf" 
    filepath = os.path.join(account_dir, filename)

    doc = SimpleDocTemplate(filepath, pagesize=landscape(letter), 
                            rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)

    # Custom colors
    primary_color = HexColor("#2a5e35")
    secondary_color = HexColor("#E2E2E2")

    # Convert HexColor to a format matplotlib can understand
    primary_color_rgb = primary_color.rgb()
    primary_color_matplotlib = (primary_color_rgb[0]/255, primary_color_rgb[1]/255, primary_color_rgb[2]/255)

    elements = []
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Center', alignment=1, textColor=primary_color, fontSize=15))

    # Title
    title_style = ParagraphStyle(
        'Title',
        parent=styles['Title'],
        fontSize=24,
        textColor=primary_color,
        spaceAfter=12
    )
    elements.append(Paragraph(f"Daily Trading Report - {report_data['date']}", title_style))
    elements.append(Paragraph(f"{report_data['account_name'].upper()}", styles['Center']))
    elements.append(Spacer(1, 0.25*inch))

    # Account Summary
    elements.append(Paragraph("1. Account Summary", styles['Heading2']))
    account_summary = [
        ["Current Equity", f"{report_data['equity']:.2f} USDT"],
        ["Equity in BTC", f"{report_data['equity_btc']:.6f} BTC"],
        ["Open Positions", str(len(report_data['open_positions']))],
        ["Trades This Week", str(report_data['last_x_days_trades'])],
    ]
    account_table = Table(account_summary, colWidths=[2*inch, 2*inch])
    account_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), secondary_color),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('TOPPADDING', (0, 0), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, primary_color)
    ]))
    elements.append(account_table)
    elements.append(Spacer(1, 0.25*inch))

    # Long-Short Analysis
    elements.append(Paragraph("2. Long-Short Analysis", styles['Heading2']))

    long_short_analysis = [
        ["Long Positions", f"{report_data['long_positions']} ({report_data['long_ratio']:.2f}%)"],
        ["Short Positions", f"{report_data['short_positions']} ({report_data['short_ratio']:.2f}%)"],
        ["Long Exposure", f"{report_data['long_exposure']:.2f} USDT"],
        ["Short Exposure", f"{report_data['short_exposure']:.2f} USDT"],
        ["Net Exposure", f"{report_data['overall_exposure']:.2f} USDT"]
    ]
    ls_table = Table(long_short_analysis, colWidths=[2*inch, 2*inch])
    ls_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), secondary_color),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('TOPPADDING', (0, 0), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, primary_color)
    ]))
    elements.append(ls_table)
    elements.append(Spacer(1, 0.25*inch))

    # Open Positions
    elements.append(Paragraph("3. Open Positions", styles['Heading2']))
    if report_data['open_positions']:
        open_positions_data = [["Symbol", "Side", "Exposure", "Entry Price", "Mark Price", "Unrealized PNL", "Leverage"]]
        for position in report_data['open_positions']:
            open_positions_data.append([
                position["symbol"],
                ("Long" if position["side"] == "Buy" else "Short"),
                f"{position['exposure']:.2f} USDT",
                f"{position['entry_price']:.4f}",
                f"{position['mark_price']:.4f}",
                f"{position['unrealized_pnl']:.2f} USDT",
                position["leverage"]
            ])
        open_positions_table = Table(open_positions_data)
        open_positions_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), primary_color),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), secondary_color),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            ('TOPPADDING', (0, 1), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        elements.append(open_positions_table)
    else:
        elements.append(Paragraph("\nNo open positions.\n", styles['Normal']))
    elements.append(Spacer(1, 0.25*inch))

    # Equity Curve
    elements.append(Paragraph("4. Equity Curve", styles['Heading2']))
    equity_curve_img = create_equity_curve_plot(report_data['account_name'])
    elements.append(Image(equity_curve_img, width=8*inch, height=4*inch))
    elements.append(Paragraph("\nThis includes deposit and withdrawals.\n", styles['Normal']))

    # Metrics section
    elements.append(Paragraph("5. Weekly Metrics", styles['Heading2']))
    fees_data = [
        ["Metric", "Amount (USDT)"],
        ["N. Trades", f"{report_data['last_x_days_trades']}"],
        ["Volume", f"{report_data['last_x_days_volume']:.2f}"],
        ["Funding Fees", f"{report_data['last_x_days_funding_fees']:.2f}"],
        ["Trading Fees", f"{report_data['last_x_days_trading_fees']:.2f}"],
        ["Total Fees", f"{report_data['last_x_days_total_fees']:.2f}"]
    ]
    fees_table = Table(fees_data, colWidths=[2*inch, 2*inch])
    fees_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), primary_color),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), secondary_color),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('TOPPADDING', (0, 1), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    elements.append(fees_table)
    elements.append(Paragraph("\nWhen the fees are negative it means it is money received.\n", styles['Normal']))

    # Add Weekly Comparison section
    elements.append(Paragraph("6. Weekly Comparison", styles['Heading2']))
    weekly_comparison_data = [
        ["Metric", "Value"],
        ["Previous Week Equity", f"{report_data['previous_week_equity_usdt']:.2f} USDT" if report_data['previous_week_equity_usdt'] is not None else "N/A"],
        ["Equity Difference", f"{report_data['equity_difference_usdt']:.2f} %" if report_data['equity_difference_usdt'] is not None else "N/A"],
        ["Previous Week Equity (BTC)", f"{report_data['previous_week_equity_btc']:.6f} BTC" if report_data['previous_week_equity_btc'] is not None else "N/A"],
        ["Equity Difference (BTC)", f"{report_data['equity_difference_btc']:.2f} %" if report_data['equity_difference_btc'] is not None else "N/A"],
    ]
    weekly_comparison_table = Table(weekly_comparison_data, colWidths=[3*inch, 2*inch])
    weekly_comparison_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), primary_color),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), secondary_color),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('TOPPADDING', (0, 1), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    elements.append(weekly_comparison_table)
    elements.append(Paragraph("\nThis isn't calculating the eventual deposit and withdrawals, so the metric is probably not accurate.\n", styles['Normal']))

    # Add new section for Adjusted Returns Performance Graph
    elements.append(Paragraph("7. Adjusted Returns Performance", styles['Heading2']))
    
    # Create the performance graph
    if 'adjusted_returns' in report_data and len(report_data['adjusted_returns']) >= 2:
        plt.figure(figsize=(10, 5))
        dates = [datetime.datetime.strptime(data['date'], '%Y-%m-%d') for data in report_data['adjusted_returns']]
        daily_returns = [data['daily_return'] for data in report_data['adjusted_returns']]
        
        # Calculate normalized cumulative returns
        normalized_returns = [100]
        for daily_return in daily_returns[1:]:  # Skip the first day as it's the baseline
            normalized_returns.append(normalized_returns[-1] * (1 + daily_return))
        
        plt.plot(dates, normalized_returns, marker='o', linestyle='-', color=primary_color_matplotlib)
        plt.title('Normalized Adjusted Returns Performance')
        plt.xlabel('Date')
        plt.ylabel('Normalized Returns (Starting at 100)')
        plt.grid(True)
        
        # Format x-axis to show dates nicely
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gcf().autofmt_xdate()  # Rotate and align the tick labels
        
        # Add horizontal line at y=100 to show the starting point
        plt.axhline(y=100, color='r', linestyle='--', alpha=0.5)
        
        # Save plot to a BytesIO object
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
        img_buffer.seek(0)
        plt.close()
        
        # Add the image to the PDF
        img = Image(img_buffer)
        img.drawHeight = 4*inch
        img.drawWidth = 7*inch
        elements.append(img)
        
        # Add a brief explanation
        elements.append(Spacer(1, 0.25*inch))
        elements.append(Paragraph("\nThe graph above shows the normalized performance of the account based on the adjusted returns. The starting value is set to 100, and subsequent values represent the cumulative growth or decline of the initial investment.\n", styles['Normal']))
        
        # Add a table with some key metrics
        total_return = (normalized_returns[-1] / 100 - 1) * 100  # Percentage change from 100
        avg_daily_return = statistics.mean(daily_returns) * 100
        performance_data = [
            ["Metric", "Value"],
            ["Starting Value", "100"],
            ["Ending Value", f"{normalized_returns[-1]:.2f}"],
            ["Total Return", f"{total_return:.2f}%"],
            ["Avg Daily Return", f"{avg_daily_return:.2f}%"]
        ]
        performance_table = Table(performance_data, colWidths=[3*inch, 2*inch])
        performance_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), primary_color),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), secondary_color),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('TOPPADDING', (0, 1), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        elements.append(performance_table)
    else:
        elements.append(Paragraph("\nInsufficient data available to generate the adjusted returns performance graph. This could be due to a lack of data points in the specified date range.\n", styles['Normal']))

    doc.build(elements)
    print(f"Report exported to {filepath}")

    return filepath

def generate_report_for_account(account):
    try:
        session = HTTP(
            api_key=account["api_key"],
            api_secret=account["api_secret"]
        )

        x_days = 7
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        open_positions = get_all_open_positions(session)
        trades_today = get_all_trades_today(session)
        processed_trades = determine_trade_action(trades_today, open_positions)
        equity = calculate_equity(session)
        exposure_data = calculate_exposures_and_ratios(open_positions)
        fees = process_fees_and_volume(session)
        equity_in_btc = equity_btc(session, equity)
        adjusted_returns = calculate_adjusted_returns(account["name"])

        previous_week_equity = get_previous_week_equity(account["name"])
        if previous_week_equity:
            equity_difference_usdt = (equity / previous_week_equity["equity_usdt"] - 1) * 100
            equity_difference_btc = (equity_in_btc / previous_week_equity["equity_btc"] - 1) * 100
        else:
            equity_difference_usdt = None
            equity_difference_btc = None

        metrics = get_last_x_days_metrics(account["name"], x_days)

        # Fetch existing deposit and withdrawal data
        conn = sqlite3.connect('database.db')
        cursor = conn.cursor()
        cursor.execute('''
        SELECT deposit, withdrawal FROM daily_reports
        WHERE date = ? AND account_name = ?
        ''', (today, account["name"]))
        existing_data = cursor.fetchone()
        conn.close()

        deposit = existing_data[0] if existing_data else 0
        withdrawal = existing_data[1] if existing_data else 0

        report_data = {
            "account_name": account["name"],
            "date": today,
            "equity": equity,
            "equity_btc": equity_in_btc,
            "open_positions": [
                {
                    "symbol": position["symbol"],
                    "side": position["side"],
                    "exposure": float(position["size"]) * float(position["markPrice"]),
                    "entry_price": float(position["avgPrice"]),
                    "mark_price": float(position["markPrice"]),
                    "unrealized_pnl": float(position["unrealisedPnl"]),
                    "leverage": position["leverage"]
                }
                for position in open_positions
            ],
            "trades_today": len(processed_trades),
            "funding_fees": fees['funding_fees'],
            "trading_fees": fees['trading_fees'],
            "total_fees": fees['total_fees'],
            'total_volume': fees['total_volume'],
            "long_positions": exposure_data['long_positions'],
            "short_positions": exposure_data['short_positions'],
            "long_exposure": exposure_data['long_exposure'],
            "short_exposure": exposure_data['short_exposure'],
            "long_ratio": exposure_data['long_ratio'],
            "short_ratio": exposure_data['short_ratio'],
            "overall_exposure": exposure_data['overall_exposure'],
            "trades": processed_trades,
            "previous_week_equity_usdt": previous_week_equity['equity_usdt'] if previous_week_equity else None,
            "equity_difference_usdt": equity_difference_usdt,
            "previous_week_equity_btc": previous_week_equity['equity_btc'] if previous_week_equity else None,
            "equity_difference_btc": equity_difference_btc,
            "last_x_days_funding_fees": metrics['funding_fees'],
            "last_x_days_trading_fees": metrics['trading_fees'],            
            "last_x_days_total_fees": metrics['total_fees'],
            "last_x_days_trades": metrics['trades'],
            "last_x_days_volume": metrics['volume'],
            "deposit": deposit,
            "withdrawal": withdrawal,
            "adjusted_returns": adjusted_returns,
        }
        
        store_daily_report(report_data)
        
        return report_data
    except Exception as e:
        logging.error(f"Error generating report for account {account['name']}: {str(e)}")
        raise

# DATA 
def collect_daily_data():
    try:
        initialize_database()
        accounts = get_accounts_from_env()
        
        for account in accounts:
            logging.info(f"Collecting daily data for {account['name']}...")
            report_data = generate_report_for_account(account)
            store_daily_report(report_data)

        logging.info("Daily data collection completed successfully.")
    except Exception as e:
        logging.error(f"Error in daily data collection: {str(e)}")
        print(f"An error occurred. Please check the log file for details.")

def generate_weekly_report():
    try:
        accounts = get_accounts_from_env()
        
        for account in accounts:
            logging.info(f"Generating weekly report for {account['name']}...")
            try:
                weekly_report_data = generate_report_for_account(account)
                
                # Calculate adjusted returns for the past 28 days instead of 7
                end_date = datetime.datetime.now().strftime("%Y-%m-%d")
                start_date = (datetime.datetime.now() - datetime.timedelta(days=28)).strftime("%Y-%m-%d")
                logging.info(f"Calculating adjusted returns for {account['name']} from {start_date} to {end_date}")
                adjusted_returns = calculate_adjusted_returns(account['name'], start_date, end_date)
                
                if adjusted_returns:
                    logging.info(f"Calculated {len(adjusted_returns)} adjusted return data points for {account['name']}")
                else:
                    logging.warning(f"No adjusted return data calculated for {account['name']}")
                
                weekly_report_data['adjusted_returns'] = adjusted_returns
                
                pdf_path = export_report_to_pdf(weekly_report_data)
                logging.info(f"PDF report generated for {account['name']}: {pdf_path}")
                # send_email_with_attachment(account['name'], account['email'], pdf_path)
            except Exception as e:
                logging.error(f"Error processing report for account {account['name']}: {str(e)}")
                continue
            
        logging.info("Weekly reports generated and sent successfully.")
    except Exception as e:
        logging.error(f"Error in weekly report generation: {str(e)}")
        print(f"An error occurred. Please check the log file for details.")


def main():
    parser = argparse.ArgumentParser(description='Trading Report Generator')
    parser.add_argument('--force-weekly', action='store_true', help='Force generate weekly report')
    args = parser.parse_args()

    # Process transactions from CSV
    process_transactions_csv()
    
    # Always run daily data collection
    collect_daily_data()
    daily_combined_report()

    # Check if it's Sunday (weekday() returns 6 for Sunday) or if --force-weekly flag is used
    # to change the day bare in mind: 0=monday, 6=sunday, ecc
    if datetime.datetime.now().weekday() == 6 or args.force_weekly:
        generate_weekly_report()
        generate_combined_report()


if __name__ == "__main__":
    main()