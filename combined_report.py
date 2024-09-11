import sqlite3
import matplotlib.pyplot as plt
from io import BytesIO
from reportlab.lib.pagesizes import letter, landscape
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_CENTER
import datetime
import os
import numpy as np
import logging
from functools import wraps

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Set up logging
logging.basicConfig(filename='trading_report.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def log_errors(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.error(f"Error in {func.__name__}: {str(e)}")
            raise
    return wrapper

os.chdir(os.path.dirname(os.path.abspath(__file__)))

@log_errors
def get_all_accounts():
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT account_name FROM daily_reports")
    accounts = [row[0] for row in cursor.fetchall()]
    conn.close()
    return accounts

@log_errors
def get_latest_data_for_accounts():
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    accounts = get_all_accounts()
    latest_data = {}
    for account in accounts:
        cursor.execute("""
            SELECT * FROM daily_reports 
            WHERE account_name = ? 
            ORDER BY date DESC 
            LIMIT 1
        """, (account,))
        data = cursor.fetchone()
        if data:
            latest_data[account] = {
                "date": data[0],
                "equity": data[2],
                "trades_today": data[5],
                "long_positions": data[6],
                "short_positions": data[7],
                "long_exposure": data[8],
                "short_exposure": data[9],
                "net_exposure": data[10],
                "funding_fees": data[11],
                "trading_fees": data[12],
                "total_fees": data[13],
                "total_volume": data[14]
            }
    conn.close()
    return latest_data

@log_errors
def create_combined_equity_curve_plot(days=30):
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    accounts = get_all_accounts()
    
    plt.figure(figsize=(10, 5))
    
    for account in accounts:
        query = '''
        SELECT date, equity
        FROM daily_reports
        WHERE account_name = ?
        ORDER BY date DESC
        LIMIT ?
        '''
        cursor.execute(query, (account, days))
        rows = cursor.fetchall()
        dates, equity = zip(*[(datetime.datetime.strptime(row[0], '%Y-%m-%d'), row[1]) for row in reversed(rows)])
        
        normalized_equity = [value / equity[0] * 100 for value in equity]
        
        plt.plot(dates, normalized_equity, marker='o', label=account)

    plt.title('Normalized Combined Equity Curve')
    plt.xlabel('Date')
    plt.ylabel('Normalized Equity (%)')
    plt.legend()
    plt.grid(True)
    plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
    plt.gcf().autofmt_xdate()
    plt.gca().yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter())
    plt.axhline(y=100, color='r', linestyle='--', alpha=0.5)

    img_buffer = BytesIO()
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    plt.close()

    conn.close()
    return img_buffer

@log_errors
def create_total_equity_curve_plot(days=30):
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    accounts = get_all_accounts()
    
    plt.figure(figsize=(10, 5))
    
    total_equity = {}
    
    for account in accounts:
        query = '''
        SELECT date, equity
        FROM daily_reports
        WHERE account_name = ?
        ORDER BY date DESC
        LIMIT ?
        '''
        cursor.execute(query, (account, days))
        rows = cursor.fetchall()
        for date, equity in reversed(rows):
            date = datetime.datetime.strptime(date, '%Y-%m-%d')
            if date in total_equity:
                total_equity[date] += equity
            else:
                total_equity[date] = equity

    dates = sorted(total_equity.keys())
    equities = [total_equity[date] for date in dates]
    
    plt.plot(dates, equities, marker='o', label='Total Equity', color='#2a5e35')

    plt.title('Total Equity Curve')
    plt.xlabel('Date')
    plt.ylabel('Total Equity (USDT)')
    plt.legend()
    plt.grid(True)
    plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
    plt.gcf().autofmt_xdate()

    img_buffer = BytesIO()
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    plt.close()

    conn.close()
    return img_buffer

@log_errors
def get_x_day_fees_and_volumes(x_days=7):
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    accounts = get_all_accounts()
    
    x_days_ago = (datetime.datetime.now() - datetime.timedelta(days=x_days)).strftime('%Y-%m-%d')
    
    fees_and_volumes = {}
    
    for account in accounts:
        cursor.execute("""
            SELECT 
                SUM(funding_fees) as total_funding_fees,
                SUM(trading_fees) as total_trading_fees,
                SUM(total_fees) as total_fees,
                SUM(total_volume) as total_volume
            FROM daily_reports 
            WHERE account_name = ? AND date >= ?
        """, (account, x_days_ago))
        
        result = cursor.fetchone()
        fees_and_volumes[account] = {
            "funding_fees": result[0] or 0,
            "trading_fees": result[1] or 0,
            "total_fees": result[2] or 0,
            "total_volume": result[3] or 0
        }
    
    conn.close()
    return fees_and_volumes

@log_errors
def get_weekly_performance():
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    accounts = get_all_accounts()
    
    today = datetime.date.today()
    one_week_ago = today - datetime.timedelta(days=7)
    
    performance = {}
    
    for account in accounts:
        cursor.execute("""
            SELECT date, equity
            FROM daily_reports 
            WHERE account_name = ? AND date IN (?, ?)
            ORDER BY date ASC
        """, (account, one_week_ago.strftime('%Y-%m-%d'), today.strftime('%Y-%m-%d')))
        
        results = cursor.fetchall()
        if len(results) == 2:
            start_date, start_equity = results[0]
            end_date, end_equity = results[1]
            percent_change = ((end_equity - start_equity) / start_equity) * 100
            performance[account] = {
                'start_equity': start_equity,
                'end_equity': end_equity,
                'percent_change': percent_change
            }
        else:
            performance[account] = None
    
    conn.close()
    return performance

@log_errors
def get_performance(days=1):
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    accounts = get_all_accounts()
    
    today = datetime.date.today()
    start_date = today - datetime.timedelta(days=days)
    
    performance = {}
    
    for account in accounts:
        cursor.execute("""
            SELECT date, equity
            FROM daily_reports 
            WHERE account_name = ? AND date IN (?, ?)
            ORDER BY date ASC
        """, (account, start_date.strftime('%Y-%m-%d'), today.strftime('%Y-%m-%d')))
        
        results = cursor.fetchall()
        if len(results) == 2:
            start_date, start_equity = results[0]
            end_date, end_equity = results[1]
            percent_change = ((end_equity - start_equity) / start_equity) * 100
            performance[account] = {
                'start_date': start_date,
                'end_date': end_date,
                'start_equity': start_equity,
                'end_equity': end_equity,
                'percent_change': percent_change
            }
        else:
            performance[account] = None
    
    conn.close()
    return performance

@log_errors
def create_table(data, title, styles):
    table = Table(data, colWidths=[2*inch, 1.5*inch, 1.5*inch, 1.5*inch, 1.5*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor("#2a5e35")),
        ('TEXTCOLOR', (0, 0), (-1, 0), HexColor("#FFFFFF")),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), HexColor("#EEEEEE")),
        ('TEXTCOLOR', (0, 1), (-1, -1), HexColor("#000000")),
        ('ALIGN', (0, 1), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('TOPPADDING', (0, 1), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 1, HexColor("#000000"))
    ]))
    return [Paragraph(title, styles['Heading2']), table, Spacer(1, 0.25*inch)]

@log_errors
def generate_combined_report(x_day=7):
    x_day_fees_and_volumes = get_x_day_fees_and_volumes(x_day)
    latest_data = get_latest_data_for_accounts()
    weekly_performance = get_weekly_performance()
    
    reports_dir = 'reports'
    os.makedirs(reports_dir, exist_ok=True)

    account_dir = os.path.join(reports_dir, 'total')
    os.makedirs(account_dir, exist_ok=True)

    today = datetime.datetime.now().strftime("%Y-%m-%d")
    filename = f"combined_report_{today}.pdf"
    filepath = os.path.join(account_dir, filename)

    doc = SimpleDocTemplate(filepath, pagesize=landscape(letter),
                            rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=18)

    elements = []
    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(name='CenteredHeading1', parent=styles['Heading1'], alignment=TA_CENTER))

    elements.append(Paragraph(f"Combined Trading Report - {today}", styles['CenteredHeading1']))
    elements.append(Spacer(1, 0.5*inch))

    account_summary_data = [["Account", "Equity (USDT)", "Long Positions", "Short Positions", "Net Exposure"]]
    total_equity_usdt = 0
    total_long_positions = 0
    total_short_positions = 0
    total_net_exposure = 0

    for account, data in latest_data.items():
        account_summary_data.append([
            account,
            f"{data['equity']:.2f}",
            str(data['long_positions']),
            str(data['short_positions']),
            f"{data['net_exposure']:.2f}"
        ])
        total_equity_usdt += data['equity']
        total_long_positions += data['long_positions']
        total_short_positions += data['short_positions']
        total_net_exposure += data['net_exposure']

    account_summary_data.append([
        "Total",
        f"{total_equity_usdt:.2f}",
        str(total_long_positions),
        str(total_short_positions),
        f"{total_net_exposure:.2f}"
    ])

    elements.extend(create_table(account_summary_data, "Account Summary", styles))

    fees_volumes_data = [["Account", "Funding Fees", "Trading Fees", "Total Fees", "Total Volume"]]
    total_funding_fees = 0
    total_trading_fees = 0
    total_fees = 0
    total_volume = 0

    for account, data in x_day_fees_and_volumes.items():
        fees_volumes_data.append([
            account,
            f"{data['funding_fees']:.2f}",
            f"{data['trading_fees']:.2f}",
            f"{data['total_fees']:.2f}",
            f"{data['total_volume']:.2f}"
        ])
        total_funding_fees += data['funding_fees']
        total_trading_fees += data['trading_fees']
        total_fees += data['total_fees']
        total_volume += data['total_volume']

    fees_volumes_data.append([
        "Total",
        f"{total_funding_fees:.2f}",
        f"{total_trading_fees:.2f}",
        f"{total_fees:.2f}",
        f"{total_volume:.2f}"
    ])

    elements.extend(create_table(fees_volumes_data, f"{x_day}-Day Fees and Volumes", styles))

    performance_data = [["Account", "Start Equity", "End Equity", "Performance", "Weight"]]
    total_start_equity = 0
    total_end_equity = 0
    total_weighted_performance = 0

    for account, data in weekly_performance.items():
        if data is not None:
            weight = data['start_equity'] / sum(d['start_equity'] for d in weekly_performance.values() if d is not None)
            performance_data.append([
                account,
                f"{data['start_equity']:.2f}",
                f"{data['end_equity']:.2f}",
                f"{data['percent_change']:.2f}%",
                f"{weight:.2%}"
            ])
            total_start_equity += data['start_equity']
            total_end_equity += data['end_equity']
            total_weighted_performance += data['percent_change'] * weight

    if total_start_equity > 0:
        total_performance = ((total_end_equity - total_start_equity) / total_start_equity) * 100
        performance_data.append([
            "Total",
            f"{total_start_equity:.2f}",
            f"{total_end_equity:.2f}",
            f"{total_performance:.2f}%",
            "100.00%"
        ])
        performance_data.append([
            "Weighted Average",
            "",
            "",
            f"{total_weighted_performance:.2f}%",
            ""
        ])
    else:
        performance_data.extend([
            ["Total", "N/A", "N/A", "N/A", "N/A"],
            ["Weighted Average", "", "", "N/A", ""]
        ])

    elements.extend(create_table(performance_data, "Weekly Performance", styles))

    elements.append(Paragraph("Normalized Combined Equity Curve", styles['Heading2']))
    equity_curve_img = create_combined_equity_curve_plot()
    elements.append(Image(equity_curve_img, width=9*inch, height=4.5*inch))
    elements.append(Spacer(1, 0.25*inch))

    elements.append(Paragraph("Total Equity Curve", styles['Heading2']))
    total_equity_curve_img = create_total_equity_curve_plot()
    elements.append(Image(total_equity_curve_img, width=9*inch, height=4.5*inch))

    try:
        doc.build(elements)
        logging.info(f"Combined report exported to {filepath}")
        return filepath
    except Exception as e:
        logging.error(f"Error building PDF document: {str(e)}")
        raise

@log_errors
def daily_combined_report(x_day=1):
    x_day_fees_and_volumes = get_x_day_fees_and_volumes(x_day)
    latest_data = get_latest_data_for_accounts()
    performance = get_performance(x_day)
    
    reports_dir = 'reports'
    os.makedirs(reports_dir, exist_ok=True)

    account_dir = os.path.join(reports_dir, 'total')
    os.makedirs(account_dir, exist_ok=True)

    today = datetime.datetime.now().strftime("%Y-%m-%d")
    filename = f"daily_combined_{today}.pdf"
    filepath = os.path.join(account_dir, filename)

    doc = SimpleDocTemplate(filepath, pagesize=landscape(letter),
                            rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=18)

    elements = []
    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(name='CenteredHeading1', parent=styles['Heading1'], alignment=TA_CENTER))

    elements.append(Paragraph(f"Daily Combined Report - {today}", styles['CenteredHeading1']))
    elements.append(Spacer(1, 0.5*inch))

    account_summary_data = [["Account", "Equity (USDT)", "Long Positions", "Short Positions", "N. Trades", "Net Exposure"]]
    total_equity_usdt = 0
    total_long_positions = 0
    total_short_positions = 0
    total_trades_today = 0
    total_net_exposure = 0

    for account, data in latest_data.items():
        account_summary_data.append([
            account,
            f"{data['equity']:.2f}",
            str(data['long_positions']),
            str(data['short_positions']),
            int(data['trades_today']),
            f"{data['net_exposure']:.2f}"
        ])
        total_equity_usdt += data['equity']
        total_long_positions += data['long_positions']
        total_short_positions += data['short_positions']
        total_trades_today += data['trades_today']
        total_net_exposure += data['net_exposure']

    account_summary_data.append([
        "Total",
        f"{total_equity_usdt:.2f}",
        str(total_long_positions),
        str(total_short_positions),
        int(total_trades_today),
        f"{total_net_exposure:.2f}"
    ])

    elements.extend(create_table(account_summary_data, "Account Summary", styles))

    fees_volumes_data = [["Account", "Funding Fees", "Trading Fees", "Total Fees", "Total Volume"]]
    total_funding_fees = 0
    total_trading_fees = 0
    total_fees = 0
    total_volume = 0

    for account, data in x_day_fees_and_volumes.items():
        fees_volumes_data.append([
            account,
            f"{data['funding_fees']:.2f}",
            f"{data['trading_fees']:.2f}",
            f"{data['total_fees']:.2f}",
            f"{data['total_volume']:.2f}"
        ])
        total_funding_fees += data['funding_fees']
        total_trading_fees += data['trading_fees']
        total_fees += data['total_fees']
        total_volume += data['total_volume']

    fees_volumes_data.append([
        "Total",
        f"{total_funding_fees:.2f}",
        f"{total_trading_fees:.2f}",
        f"{total_fees:.2f}",
        f"{total_volume:.2f}"
    ])

    elements.extend(create_table(fees_volumes_data, f"{x_day}-Day Fees and Volumes", styles))

    performance_data = [["Account", "Start Equity", "End Equity", "Performance", "Weight"]]
    total_start_equity = 0
    total_end_equity = 0
    total_weighted_performance = 0

    for account, data in performance.items():
        if data is not None:
            weight = data['start_equity'] / sum(d['start_equity'] for d in performance.values() if d is not None)
            performance_data.append([
                account,
                f"{data['start_equity']:.2f}",
                f"{data['end_equity']:.2f}",
                f"{data['percent_change']:.2f}%",
                f"{weight:.2%}"
            ])
            total_start_equity += data['start_equity']
            total_end_equity += data['end_equity']
            total_weighted_performance += data['percent_change'] * weight

    if total_start_equity > 0:
        total_performance = ((total_end_equity - total_start_equity) / total_start_equity) * 100
        performance_data.append([
            "Total",
            f"{total_start_equity:.2f}",
            f"{total_end_equity:.2f}",
            f"{total_performance:.2f}%",
            "100.00%"
        ])
        performance_data.append([
            "Weighted Average",
            "",
            "",
            f"{total_weighted_performance:.2f}%",
            ""
        ])
    else:
        performance_data.extend([
            ["Total", "N/A", "N/A", "N/A", "N/A"],
            ["Weighted Average", "", "", "N/A", ""]
        ])

    elements.extend(create_table(performance_data, "Yesterday's Performance", styles))

    elements.append(Paragraph("Normalized Combined Equity Curve", styles['Heading2']))
    equity_curve_img = create_combined_equity_curve_plot()
    elements.append(Image(equity_curve_img, width=9*inch, height=4.5*inch))
    elements.append(Spacer(1, 0.25*inch))

    elements.append(Paragraph("Total Equity Curve", styles['Heading2']))
    total_equity_curve_img = create_total_equity_curve_plot()
    elements.append(Image(total_equity_curve_img, width=9*inch, height=4.5*inch))

    try:
        doc.build(elements)
        logging.info(f"Combined report exported to {filepath}")
        return filepath
    except Exception as e:
        logging.error(f"Error building PDF document: {str(e)}")
        raise


if __name__ == "__main__":
    try:
        logging.info("Starting combined report generation...")
        # report_path = generate_combined_report()
        daily_combined_report()
    except Exception as e:
        logging.error(f"Error generating combined report: {str(e)}")
        print(f"An error occurred while generating the report. Please check the log file for details.")