from pybit.unified_trading import HTTP
from dotenv import dotenv_values
from datetime import datetime, timedelta
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
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from io import BytesIO
from decimal import Decimal
import logging
import time
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import colorsys
import argparse
from main2 import get_accounts_from_env, get_all_open_positions

os.chdir(os.path.dirname(os.path.abspath(__file__)))

db_path = 'db/database.db'

# Set up logging
logging.basicConfig(filename='positions_analysis.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def get_unique_color(index, total):
    """Generate unique color using HSV color space for better distinction."""
    hue = index / total
    saturation = 0.7
    value = 0.85
    rgb = colorsys.hsv_to_rgb(hue, saturation, value)
    return rgb

def get_positions(account):
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

        print(f"account: {account['name']}")
        print(open_positions)
        print("-------------------------")

    except Exception as e:
        logging.error(f"Error generating report for account {account['name']}: {str(e)}")
        raise

def get_historical_data(session, symbols, days=7, cache_dir='data'):
    """
    Fetch historical data for VaR calculation with caching
    """
    interval = "60"
    try:
        os.makedirs(cache_dir, exist_ok=True)
        all_data = {}
        current_time = datetime.now()
        
        for symbol in symbols:
            cache_file = os.path.join(cache_dir, f"{symbol}_{interval}_{days}d.json")
            
            # Try to use cached data
            if os.path.exists(cache_file):
                with open(cache_file, 'r') as f:
                    cached_data = json.load(f)
                    last_updated = datetime.fromtimestamp(cached_data['last_updated'])
                    
                    if current_time - last_updated < timedelta(hours=1):
                        # logger.info(f"Using cached data for {symbol}")
                        df = pd.DataFrame(cached_data['data'])
                        
                        # Handle timestamp based on format
                        try:
                            # If timestamp is milliseconds
                            if df['timestamp'].iloc[0].isdigit():
                                df['timestamp'] = pd.to_datetime(df['timestamp'].astype(float), unit='ms')
                            else:
                                # If timestamp is already datetime string
                                df['timestamp'] = pd.to_datetime(df['timestamp'])
                        except Exception as e:
                            logging.error(f"Error converting cached timestamps for {symbol}: {e}")
                            continue
                            
                        all_data[symbol] = df
                        continue
            
            # Fetch new data from API
            logging.info(f"Fetching new data for {symbol}")
            try:
                response = session.get_kline(
                    category="linear",
                    symbol=symbol,
                    interval=interval,
                    limit=days * 24
                )
                
                if response['retCode'] == 0:
                    kline_data = response['result']['list']
                    df = pd.DataFrame(kline_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
                    
                    # Convert numeric columns first
                    numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'turnover']
                    for col in numeric_columns:
                        df[col] = df[col].astype(float)
                    
                    # Convert timestamp to datetime explicitly from milliseconds
                    df['timestamp'] = pd.to_datetime(pd.to_numeric(df['timestamp']), unit='ms')
                    
                    # Store in all_data
                    all_data[symbol] = df
                    
                    # Prepare data for caching
                    cache_data = {
                        'last_updated': current_time.timestamp(),
                        'data': df.assign(timestamp=df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')).to_dict('records')
                    }
                    
                    # Save to cache file
                    with open(cache_file, 'w') as f:
                        json.dump(cache_data, f)
                    logging.info(f"Cached data for {symbol}")
                else:
                    logging.error(f"Error fetching data for {symbol}: {response['retMsg']}")
                    
            except Exception as api_error:
                logging.error(f"API error for {symbol}: {api_error}")
                continue
        
        return all_data
        
    except Exception as e:
        logging.error(f"Error in get_historical_data: {e}")
        # Try to recover using cached data
        try:
            all_data = {}
            for symbol in symbols:
                cache_file = os.path.join(cache_dir, f"{symbol}_{interval}_{days}d.json")
                if os.path.exists(cache_file):
                    with open(cache_file, 'r') as f:
                        cached_data = json.load(f)
                        df = pd.DataFrame(cached_data['data'])
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                        all_data[symbol] = df
                        logging.info(f"Recovered using cached data for {symbol}")
            return all_data if all_data else None
        except Exception as cache_error:
            logging.error(f"Error reading cache: {cache_error}")
            return None

def analyze_trade_performance(account, days=7):
    """
    Analyze trade performance and direction for a specific account.
    Performance is normalized to start from zero.
    
    Args:
        account (dict): Account credentials and information
        days (int): Number of days to analyze
    """
    try:
        # Initialize API session
        session = HTTP(
            api_key=account["api_key"],
            api_secret=account["api_secret"]
        )

        # Get open positions
        open_positions = get_all_open_positions(session)
        
        if not open_positions:
            logging.info(f"No open positions found for account {account['name']}")
            return
            
        # Get symbols from open positions
        symbols = [position['symbol'] for position in open_positions]
        
        # Fetch historical data for all symbols
        historical_data = get_historical_data(session, symbols, days=days)
        
        if not historical_data:
            logging.error("Failed to fetch historical data")
            return
            
        # Create visualization
        plt.figure(figsize=(15, 10))
        
        # Store lines and their final values for sorting
        lines_data = []
        
        # Generate unique colors for each symbol
        num_positions = len(open_positions)
        
        for idx, position in enumerate(open_positions):
            symbol = position['symbol']
            side = position['side']  # 'Buy' for long, 'Sell' for short
            size = float(position['size'])
            entry_price = float(position['avgPrice'])
            
            if symbol in historical_data:
                df = historical_data[symbol].copy()  # Create a copy to avoid SettingWithCopyWarning
                
                # Get the first close price
                first_close = df['close'].iloc[-1]  # Use the oldest price as reference
                
                # Calculate performance based on position side relative to the first price
                if side.lower() == 'buy':
                    df['performance'] = ((df['close'] - first_close) / first_close) * 100
                else:
                    df['performance'] = ((first_close - df['close']) / first_close) * 100
                
                # Reverse the data so it starts from zero
                df = df.iloc[::-1]
                
                # Get unique color for this symbol
                color = get_unique_color(idx, num_positions)
                
                # Plot and store the line with its final value
                line = plt.plot(df['timestamp'], df['performance'], 
                              label=f"{symbol} ({side})", 
                              color=color,
                              linewidth=2,
                              alpha=0.8)[0]
                
                final_value = df['performance'].iloc[-1]
                lines_data.append((line, final_value))
        
        # Sort lines by final value and reorder legend
        lines_data.sort(key=lambda x: x[1], reverse=True)
        
        # Create legend with sorted entries
        legend_labels = [f"{line.get_label()} ({value:.2f}%)" for line, value in lines_data]
        plt.legend(
            [line for line, _ in lines_data],
            legend_labels,
            bbox_to_anchor=(1.05, 1),
            loc='upper left'
        )
        
        plt.title(f"Trade Performance for {account['name']} (Last {days} Days)")
        plt.xlabel("Date")
        plt.ylabel("Performance Change (%)")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.3)  # Add horizontal line at 0
        plt.tight_layout()
        
        # Add timestamp range in the corner
        if historical_data and len(historical_data) > 0:
            first_symbol = list(historical_data.keys())[0]
            date_range = historical_data[first_symbol]['timestamp']
            start_date = date_range.iloc[-1].strftime('%Y-%m-%d')  # Reversed order
            end_date = date_range.iloc[0].strftime('%Y-%m-%d')
            plt.text(0.02, 0.02, f'Period: {start_date} to {end_date}', 
                    transform=plt.gca().transAxes, 
                    fontsize=8, 
                    alpha=0.7)
        
        # Save the plot
        output_dir = 'analysis_output'
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"trade_performance_{account['name']}_{datetime.now().strftime('%Y%m%d')}.png")
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        logging.info(f"Trade analysis completed for account {account['name']}. Plot saved to {output_path}")
        return output_path
        
    except Exception as e:
        logging.error(f"Error analyzing trade performance for account {account['name']}: {str(e)}")
        raise

def main():
    accounts = get_accounts_from_env()
    for account in accounts:
        try:
            output_path = analyze_trade_performance(account)
            if output_path:
                print(f"Analysis completed. Plot saved to: {output_path}")
        except Exception as e:
            print(f"Error analyzing account {account['name']}: {str(e)}")


if __name__ == "__main__":
    main()