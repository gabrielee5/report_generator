import sqlite3
import matplotlib.pyplot as plt
from io import BytesIO
from reportlab.lib.pagesizes import letter, landscape
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor
import datetime
import os
import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))

def get_all_accounts():
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT account_name FROM daily_reports")
    accounts = [row[0] for row in cursor.fetchall()]
    conn.close()
    return accounts

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
        
        # Normalize the equity curve
        normalized_equity = [value / equity[0] * 100 for value in equity]
        
        plt.plot(dates, normalized_equity, marker='o', label=account)

    plt.title('Normalized Combined Equity Curve')
    plt.xlabel('Date')
    plt.ylabel('Normalized Equity (%)')
    plt.legend()
    plt.grid(True)
    plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
    plt.gcf().autofmt_xdate()
    
    # Set y-axis to percentage
    plt.gca().yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter())

    # Add horizontal line at 100%
    plt.axhline(y=100, color='r', linestyle='--', alpha=0.5)

    img_buffer = BytesIO()
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    plt.close()

    conn.close()
    return img_buffer

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

def generate_combined_report():
    x_day = 7
    x_day_fees_and_volumes = get_x_day_fees_and_volumes(x_day)
    latest_data = get_latest_data_for_accounts()
    
    # Create 'reports' directory if it doesn't exist
    reports_dir = 'reports'
    os.makedirs(reports_dir, exist_ok=True)

    # Create account-specific directory
    account_dir = os.path.join(reports_dir, 'total')
    os.makedirs(account_dir, exist_ok=True)

    # Generate filename with current date
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    filename = f"combined_report_{today}.pdf"
    filepath = os.path.join(account_dir, filename)

    doc = SimpleDocTemplate(filepath, pagesize=landscape(letter),
                            rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)

    elements = []
    styles = getSampleStyleSheet()

    # Title
    title_style = getSampleStyleSheet()['Title']
    title_style.textColor = HexColor("#2a5e35")
    elements.append(Paragraph(f"Combined Trading Report - {today}", title_style))
    elements.append(Spacer(1, 0.25*inch))

    # Account Summary Table
    elements.append(Paragraph("Account Summary", styles['Heading2']))
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

    account_table = Table(account_summary_data)
    account_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor("#2a5e35")),
        ('TEXTCOLOR', (0, 0), (-1, 0), HexColor("#FFFFFF")),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
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
    elements.append(account_table)
    elements.append(Spacer(1, 0.25*inch))

    # 7-Day Fees and Volumes Table
    elements.append(Paragraph(f"{x_day}-Day Fees and Volumes", styles['Heading2']))
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

    fees_volumes_table = Table(fees_volumes_data)
    fees_volumes_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor("#2a5e35")),
        ('TEXTCOLOR', (0, 0), (-1, 0), HexColor("#FFFFFF")),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
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
    elements.append(fees_volumes_table)
    elements.append(Spacer(1, 0.25*inch))

    # Normalized Combined Equity Curve
    elements.append(Paragraph("Normalized Combined Equity Curve", styles['Heading2']))
    equity_curve_img = create_combined_equity_curve_plot()
    elements.append(Image(equity_curve_img, width=8*inch, height=4*inch))
    elements.append(Spacer(1, 0.25*inch))

    # Total Equity Curve
    elements.append(Paragraph("Total Equity Curve", styles['Heading2']))
    total_equity_curve_img = create_total_equity_curve_plot()
    elements.append(Image(total_equity_curve_img, width=8*inch, height=4*inch))

    doc.build(elements)
    print(f"Combined report exported to {filepath}")
    return filepath

if __name__ == "__main__":
    generate_combined_report()