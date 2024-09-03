from pybit.unified_trading import HTTP
from dotenv import dotenv_values
import datetime
from pybit.exceptions import InvalidRequestError
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, landscape
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
import sqlite3
from tabulate import tabulate
import os

# functioning code of the report generator for a single account to pdf 

# Load environment variables
secrets = dotenv_values(".env")
api_key = secrets["api_key"]
api_secret = secrets["api_secret"]

# Initialize Bybit client
session = HTTP(
    api_key=api_key,
    api_secret=api_secret
)


# DATABASE
def initialize_database():
    conn = sqlite3.connect('trading_bot_reports.db')
    cursor = conn.cursor()
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS daily_reports (
        date TEXT PRIMARY KEY,
        equity REAL,
        open_positions INTEGER,
        trades_today INTEGER,
        long_positions INTEGER,
        short_positions INTEGER,
        long_exposure REAL,
        short_exposure REAL,
        net_exposure REAL
    )
    ''')
    
    conn.commit()
    conn.close()

def store_daily_report(report_data):
    conn = sqlite3.connect('trading_bot_reports.db')
    cursor = conn.cursor()
    
    cursor.execute('''
    INSERT OR REPLACE INTO daily_reports 
    (date, equity, open_positions, trades_today, long_positions, short_positions, 
    long_exposure, short_exposure, net_exposure)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        report_data['date'],
        report_data['equity'],
        len(report_data['open_positions']),  # Store the count of open positions
        report_data['trades_today'],
        report_data['long_positions'],
        report_data['short_positions'],
        report_data['long_exposure'],
        report_data['short_exposure'],
        report_data['overall_exposure']
    ))
    
    conn.commit()
    conn.close()



def get_all_open_positions():
    response = session.get_positions(
        category="linear",
        settleCoin="USDT"
    )
    positions = response["result"]["list"]
    return [position for position in positions if float(position["size"]) != 0]

def parse_executions(executions, start_time):
    return [
        {
            "symbol": exec["symbol"],
            "side": exec["side"],
            "price": float(exec["execPrice"]),
            "quantity": float(exec["execQty"]),
            "value": float(exec["execValue"]),
            "fee": float(exec["execFee"]),
            "time": datetime.datetime.fromtimestamp(int(exec["execTime"]) / 1000),
            "type": exec["execType"],
            "orderId": exec["orderId"]
        }
        for exec in executions
        if exec["execType"] == "Trade" and int(exec["execTime"]) >= int(start_time.timestamp() * 1000)
    ]

def get_all_trades_today():
    today = datetime.datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    all_trades = []
    cursor = None

    while True:
        try:
            response = session.get_executions(
                category="linear",
                limit=100,
                cursor=cursor
            )

            executions = response["result"]["list"]
            parsed_executions = parse_executions(executions, today)
            all_trades.extend(parsed_executions)

            cursor = response["result"].get("nextPageCursor")
            if not cursor or not executions or (executions and int(executions[-1]["execTime"]) < int(today.timestamp() * 1000)):
                break

        except InvalidRequestError as e:
            print(f"Error fetching executions: {e}")
            break

    return all_trades

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
            action = f"Closing {"Long" if side == "Sell" else "Short"}"
        elif now_position > 0:
            action = f"Opening {"Long" if side == "Buy" else "Short"}"
        
        positions[symbol] = {"size": now_position, "side": "Long" if now_position > 0 else "Short"}
        
        processed_trade = trade.copy()
        processed_trade["action"] = action
        processed_trades.append(processed_trade)

    return processed_trades

def calculate_equity_curve():
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


def export_report_to_pdf(report_data):
    # Create 'reports' directory if it doesn't exist
    reports_dir = 'reports'
    os.makedirs(reports_dir, exist_ok=True)

    # Generate filename with current date
    filename = f"report_{report_data['date']}.pdf"
    filepath = os.path.join(reports_dir, filename)

    doc = SimpleDocTemplate(filepath, pagesize=landscape(letter))
    elements = []
    styles = getSampleStyleSheet()

    # Title
    elements.append(Paragraph(f"Daily Trading Report - {report_data['date']}", styles['Title']))

    # Account Summary
    elements.append(Paragraph("1. Account Summary", styles['Heading2']))
    account_summary = [
        ["Current Equity", f"{report_data['equity']:.2f} USDT"],
        ["Open Positions", str(len(report_data['open_positions']))],
        ["Trades Today", str(report_data['trades_today'])],
    ]
    elements.append(Table(account_summary))

    # Long-Short Analysis
    elements.append(Paragraph("2. Long-Short Analysis", styles['Heading2']))
    long_short_analysis = [
        ["Long Positions", f"{report_data['long_positions']} ({report_data['long_ratio']:.2f}%)"],
        ["Short Positions", f"{report_data['short_positions']} ({report_data['short_ratio']:.2f}%)"],
        ["Long Exposure", f"{report_data['long_exposure']:.2f} USDT"],
        ["Short Exposure", f"{report_data['short_exposure']:.2f} USDT"],
        ["Net Exposure", f"{report_data['overall_exposure']:.2f} USDT"]
    ]
    elements.append(Table(long_short_analysis))

    # Open Positions
    elements.append(Paragraph("3. Open Positions", styles['Heading2']))
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
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('TOPPADDING', (0, 1), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    elements.append(open_positions_table)

    # Today's Trades
    elements.append(Paragraph("4. Today's Trades", styles['Heading2']))
    trades_data = [["Symbol", "Action", "Price", "Quantity", "Value", "Fee", "Time"]]
    for trade in report_data['trades']:
        trades_data.append([
            trade["symbol"],
            trade["action"],
            f"{trade['price']:.4f}",
            f"{trade['quantity']:.4f}",
            f"{trade['value']:.2f}",
            f"{trade['fee']:.4f}",
            trade["time"].strftime("%Y-%m-%d %H:%M:%S")
        ])
    trades_table = Table(trades_data)
    trades_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('TOPPADDING', (0, 1), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    elements.append(trades_table)

    doc.build(elements)
    print(f"Report exported to {filepath}")


def generate_daily_report():
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    open_positions = get_all_open_positions()
    trades_today = get_all_trades_today()
    processed_trades = determine_trade_action(trades_today, open_positions)
    equity = calculate_equity_curve()
    exposure_data = calculate_exposures_and_ratios(open_positions)

    report_data = {
        "date": today,
        "equity": equity,
        "open_positions": open_positions,
        "trades_today": len(processed_trades),
        "long_positions": exposure_data['long_positions'],
        "short_positions": exposure_data['short_positions'],
        "long_ratio": exposure_data['long_ratio'],
        "short_ratio": exposure_data['short_ratio'],
        "long_exposure": exposure_data['long_exposure'],
        "short_exposure": exposure_data['short_exposure'],
        "overall_exposure": exposure_data['overall_exposure'],
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
        "trades": processed_trades
    }

    # Store the report data in the database
    store_daily_report(report_data)

    report = f"""
    Daily Trading Report
    Date: {today}

    1. Account Summary:
    -------------------
    Current Equity: {equity:.2f} USDT
    Open Positions: {len(open_positions)}
    Trades Today: {len(processed_trades)}

    2. Long-Short Analysis:
    -----------------------
    Long Positions: {exposure_data['long_positions']} ({exposure_data['long_ratio']:.2f}%)
    Short Positions: {exposure_data['short_positions']} ({exposure_data['short_ratio']:.2f}%)

    Long Exposure: {exposure_data['long_exposure']:.2f} USDT
    Short Exposure: {exposure_data['short_exposure']:.2f} USDT
    Net Exposure: {exposure_data['overall_exposure']:.2f} USDT

    3. Open Positions:
    ------------------
    """

    for position in open_positions:
        size = float(position["size"])
        mark_price = float(position["markPrice"])
        exposure = size * mark_price
        unrealized_pnl = float(position["unrealisedPnl"])
        entry_price = float(position["avgPrice"])

        report += f"""
    Symbol: {position["symbol"]}
    Side: {"Long" if position["side"] == "Buy" else "Short"}
    Exposure: {exposure:.2f} USDT
    Entry Price: {entry_price:.4f}
    Mark Price: {mark_price:.4f}
    Unrealized PNL: {unrealized_pnl:.2f} USDT
    Leverage: {position["leverage"]}
    """

    report += "\n4. Today's Trades:\n------------------\n"
    for trade in processed_trades:
        report += f"""
    Symbol: {trade["symbol"]}
    Action: {trade["action"]}
    Price: {trade["price"]}
    Quantity: {trade["quantity"]}
    Value: {trade["value"]}
    Fee: {trade["fee"]}
    Time: {trade["time"]}
    """

    print(report)

    # Export report to PDF
    export_report_to_pdf(report_data)


# to check if the database works properly
def view_stored_reports(limit=7):
    conn = sqlite3.connect('trading_bot_reports.db')
    cursor = conn.cursor()
    
    cursor.execute('''
    SELECT date, equity, open_positions, trades_today, long_positions, short_positions, 
           long_exposure, short_exposure, net_exposure
    FROM daily_reports
    ORDER BY date DESC
    LIMIT ?
    ''', (limit,))
    
    rows = cursor.fetchall()
    conn.close()
    
    if not rows:
        print("No data found in the database.")
        return
    
    headers = ['Date', 'Equity', 'Open Pos', 'Trades', 'Long Pos', 'Short Pos', 
               'Long Exp', 'Short Exp', 'Net Exp']
    
    print(tabulate(rows, headers=headers, floatfmt=".2f", tablefmt="grid"))


def main():
    initialize_database()
    generate_daily_report()

    # print("\nStored Reports (Last 7 days):")
    # view_stored_reports()

if __name__ == "__main__":
    main()