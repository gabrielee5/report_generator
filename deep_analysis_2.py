
import sqlite3
import pandas as pd
from datetime import datetime
import numpy as np

from main import get_accounts_from_env

def get_data(db_path):
    """
    Connect to the SQLite database, retrieve all data from daily_reports table,
    and close the connection in a single function.
    
    Args:
        db_path (str): Path to the SQLite database
        
    Returns:
        pandas.DataFrame: DataFrame containing all data from daily_reports table,
                         or None if an error occurs
    """
    conn = None
    try:
        # Connect to the database
        print(f"Connecting to database: {db_path}")
        conn = sqlite3.connect(db_path)
        
        # Fetch all data
        query = "SELECT * FROM daily_reports ORDER BY date"
        df = pd.read_sql_query(query, conn)
        
        # Convert date string to datetime object
        df['date'] = pd.to_datetime(df['date'])
        
        print(f"Successfully retrieved {len(df)} rows of data")
        return df
        
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None
    finally:
        # Close the connection in the finally block to ensure it happens
        # even if an exception occurs
        if conn:
            conn.close()
            print("Database connection closed")

def get_account_data(df, account):
    """
    Filter the DataFrame for a specific account and return the filtered DataFrame.
    
    Args:
        df (pandas.DataFrame): DataFrame containing all data
        account (str): Account name to filter by
        
    Returns:
        pandas.DataFrame: Filtered DataFrame for the specified account
    """
    filtered_df = df[df['account_name'] == account]
    print(f"Filtered data for account '{account}': {len(filtered_df)} rows")
    return filtered_df

def normalize_equity(df):
    """
    Normalize equity data to start from 100, adjusting for deposits and withdrawals.
    
    Args:
        df (pandas.DataFrame): DataFrame with trading data for a specific account
        
    Returns:
        pandas.DataFrame: DataFrame with normalized equity values added
    """    
    if df is None or df.empty:
        print("No data to normalize")
        return df
        
    # Make a copy to avoid modifying the original dataframe
    result_df = df.copy()
    
    # Sort data by date
    result_df.sort_values('date', inplace=True)
    result_df.reset_index(drop=True, inplace=True)
    
    # Calculate daily raw change in equity
    result_df['prev_equity'] = result_df['equity'].shift(1)
    result_df['equity_change'] = result_df['equity'] - result_df['prev_equity']
    
    # Adjust the change for deposits/withdrawals
    result_df['adjusted_change'] = result_df['equity_change'] - result_df['deposit'] + result_df['withdrawal']
    
    # Calculate daily return rate (percentage change)
    result_df['daily_return'] = result_df['adjusted_change'] / result_df['prev_equity'].replace(0, np.nan)
    
    # Start normalized equity at 100
    result_df['normalized_equity'] = 100.0
    
    # Calculate cumulative performance starting at 100
    for i in range(1, len(result_df)):
        if not np.isnan(result_df.loc[i, 'daily_return']):
            result_df.loc[i, 'normalized_equity'] = result_df.loc[i-1, 'normalized_equity'] * (1 + result_df.loc[i, 'daily_return'])
        else:
            result_df.loc[i, 'normalized_equity'] = result_df.loc[i-1, 'normalized_equity']
    
    return result_df

def fill_missing_days(normalized_df):
    """
    Fill in missing days in the normalized equity data and recalculate daily returns.
    
    Args:
        normalized_df (pandas.DataFrame): DataFrame with normalized equity values
        
    Returns:
        pandas.DataFrame: DataFrame with filled missing days and recalculated returns
    """    
    if normalized_df is None or normalized_df.empty:
        print("No data to fill")
        return normalized_df
    
    # Make a copy to avoid modifying the original dataframe
    filled_df = normalized_df.copy()
    
    # Ensure the data is sorted by date
    filled_df.sort_values('date', inplace=True)
    
    # Create a complete date range from min to max date
    min_date = filled_df['date'].min()
    max_date = filled_df['date'].max()
    complete_date_range = pd.date_range(start=min_date, end=max_date, freq='D')
    
    # Create a new DataFrame with the complete date range
    complete_df = pd.DataFrame({'date': complete_date_range})
    
    # Merge with the original data
    filled_df = pd.merge(complete_df, filled_df, on='date', how='left')
    
    # Forward fill account_name (assuming it's constant)
    if 'account_name' in filled_df.columns:
        filled_df['account_name'] = filled_df['account_name'].ffill()
    
    # For normalized_equity, we'll use linear interpolation (average of before and after)
    filled_df['normalized_equity'] = filled_df['normalized_equity'].interpolate(method='linear')
    
    # Recalculate daily returns based on the filled normalized equity values
    filled_df['prev_norm_equity'] = filled_df['normalized_equity'].shift(1)
    filled_df['filled_daily_return'] = filled_df['normalized_equity'] / filled_df['prev_norm_equity'] - 1
    
    # First day will have NaN return, set it to 0
    filled_df.loc[filled_df.index[0], 'filled_daily_return'] = 0.0
    
    print(f"Filled {len(filled_df) - len(normalized_df)} missing days in the date range")
    
    return filled_df

def analyze_filled_returns(filled_df):
    """
    Analyze the recalculated daily returns from the filled data.
    
    Args:
        filled_df (pandas.DataFrame): DataFrame with filled missing days
        
    Returns:
        dict: Dictionary with performance metrics
    """
    # Filter out rows with NaN returns
    valid_returns = filled_df.dropna(subset=['filled_daily_return'])
    
    # Basic return metrics
    avg_daily_return = valid_returns['filled_daily_return'].mean()
    annualized_return = (1 + avg_daily_return) ** 365 - 1
    
    # Risk metrics
    daily_volatility = valid_returns['filled_daily_return'].std()
    annualized_volatility = daily_volatility * np.sqrt(365)
    
    # Drawdown calculation
    returns = valid_returns['filled_daily_return']
    cumulative_returns = (1 + returns).cumprod()
    running_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns / running_max) - 1
    max_drawdown = drawdown.min()
    
    # Sharpe ratio (assuming 0% risk-free rate for simplicity)
    sharpe_ratio = (avg_daily_return / daily_volatility) * np.sqrt(365) if daily_volatility > 0 else 0
    
    # Sortino ratio (only considers downside deviation)
    downside_returns = returns[returns < 0]
    downside_deviation = downside_returns.std() if len(downside_returns) > 0 else 0
    sortino_ratio = (avg_daily_return / downside_deviation) * np.sqrt(365) if downside_deviation > 0 else 0
    
    # Win rate
    up_days = (returns > 0).mean()
    
    # Return metrics in a dictionary
    metrics = {
        'avg_daily_return': avg_daily_return,
        'annualized_return': annualized_return,
        'daily_volatility': daily_volatility,
        'annualized_volatility': annualized_volatility,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'up_days': up_days,
        'max_drawdown': max_drawdown
    }
    
    return metrics

def main():
    db = "db/database.db"
    df = get_data(db)
    if df is None:
        print("No data retrieved. Exiting.")
        return

    # get active accounts from env
    # accounts = [account['name'] for account in get_accounts_from_env()]
    accounts = ['gabriele'] 

    for account in accounts:
        account_df = get_account_data(df, account)
        normalized_equity = normalize_equity(account_df)
        filled_equity = fill_missing_days(normalized_equity)

        if filled_equity is not None:
            metrics = analyze_filled_returns(filled_equity)
            print(metrics)


if __name__ == "__main__":
    main()