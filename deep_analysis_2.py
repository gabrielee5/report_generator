import sqlite3
import pandas as pd
from datetime import datetime
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages

from main import get_accounts_from_env

PRIMARY_COLOR = '#2a5e35'  # Dark green
SECONDARY_COLOR = '#d1d1d1'  # Light gray

# GET DATA
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

# INDIVIDUAL ACCOUNT REPORT
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
    
    return filled_df

def analyze_filled_returns(filled_df):
    """
    Analyze the recalculated daily returns from the filled data.
    
    Args:
        filled_df (pandas.DataFrame): DataFrame with filled missing days
        
    Returns:
        dict: Dictionary with performance metrics including total return and days active
    """
    # Filter out rows with NaN returns
    valid_returns = filled_df.dropna(subset=['filled_daily_return'])
    
    # Calculate total return (from first to last normalized equity value)
    first_equity = filled_df['normalized_equity'].iloc[0]
    last_equity = filled_df['normalized_equity'].iloc[-1]
    total_return = (last_equity / first_equity) - 1
    
    # Calculate days active (difference between first and last date)
    first_date = filled_df['date'].min()
    last_date = filled_df['date'].max()
    days_active = (last_date - first_date).days
    
    # Basic return metrics
    avg_daily_return = valid_returns['filled_daily_return'].mean()
    annualized_return = (1 + total_return) ** (365 / days_active) - 1
    
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
        'days_active': days_active,
        'total_return': total_return,
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

def process_data(account_df):
        normalized_equity = normalize_equity(account_df)
        filled_equity = fill_missing_days(normalized_equity)
        filled_equity.to_csv('test.csv', index=False) # just for testing
        # add here some calculations if needed 
        # maybe add net_exposure here
        return filled_equity

def create_equity_curve_chart(account_df, output_path=None):
    """
    Create a chart showing the equity curve along with cumulative net deposits/withdrawals.
    The function ensures a continuous line by connecting data points even when days are missing.
    
    Args:
        account_df (pandas.DataFrame): DataFrame containing account data
        output_path (str, optional): Path to save the image file
    
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    # Make a copy to avoid modifying the original dataframe
    df = account_df.copy()
    
    # Ensure data is sorted by date
    df.sort_values('date', inplace=True)
    
    # Calculate cumulative deposits and withdrawals
    df['net_flow'] = df['deposit'] - df['withdrawal']
    df['cumulative_flow'] = df['net_flow'].cumsum()
    
    # Create a complete date range from min to max date
    min_date = df['date'].min()
    max_date = df['date'].max()
    complete_date_range = pd.date_range(start=min_date, end=max_date, freq='D')
    
    # Create a new DataFrame with the complete date range
    complete_df = pd.DataFrame({'date': complete_date_range})
    
    # Merge with the original data
    merged_df = pd.merge(complete_df, df, on='date', how='left')
    
    # Forward fill account_name (assuming it's constant)
    merged_df['account_name'] = merged_df['account_name'].ffill().bfill()
    
    # Interpolate equity and cumulative_flow values to fill gaps
    merged_df['equity'] = merged_df['equity'].interpolate(method='linear')
    merged_df['cumulative_flow'] = merged_df['cumulative_flow'].interpolate(method='linear')
    
    # Create figure and primary axis
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot equity curve - now a continuous line with primary color
    ax1.plot(merged_df['date'], merged_df['equity'], color=PRIMARY_COLOR, linestyle='-', label='Equity')
    
    # Plot cumulative net deposits/withdrawals - with secondary color
    ax1.plot(merged_df['date'], merged_df['cumulative_flow'], color=SECONDARY_COLOR, linestyle='--', 
             label='Cumulative Net Deposits/Withdrawals')
    
    # Set title and labels for primary axis
    ax1.set_title(f"Equity Curve - {merged_df['account_name'].iloc[0]}")
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Value ($)', color=PRIMARY_COLOR)
    
    # Format the x-axis to show dates clearly
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    
    # Set appropriate date locator based on the date range
    date_range = (max_date - min_date).days
    if date_range > 730:  # More than 2 years
        ax1.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 4, 7, 10)))  # Quarterly
    elif date_range > 180:  # More than 6 months
        ax1.xaxis.set_major_locator(mdates.MonthLocator())  # Monthly
    else:
        ax1.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0))  # Weekly (Mondays)
    
    plt.xticks(rotation=45)
    
    # Add grid and legend
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if output path is provided
    if output_path:
        plt.savefig(output_path)
    
    return fig

def create_normalized_equity_chart(filled_df, output_path=None):
    """
    Create a chart showing the normalized equity curve.
    
    Args:
        filled_df (pandas.DataFrame): DataFrame with normalized equity values
        output_path (str, optional): Path to save the image file
    
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot normalized equity with primary color
    ax.plot(filled_df['date'], filled_df['normalized_equity'], color=PRIMARY_COLOR, linestyle='-')
    
    # Set title and labels
    ax.set_title(f"Normalized Equity Curve - {filled_df['account_name'].iloc[0]}")
    ax.set_xlabel('Date')
    ax.set_ylabel('Normalized Value (Starting at 100)')
    
    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.xticks(rotation=45)
    
    # Add reference line at 100 with secondary color
    ax.axhline(y=100, color=SECONDARY_COLOR, linestyle='-', alpha=0.7)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if output path is provided
    if output_path:
        plt.savefig(output_path)
    
    return fig

def create_equity_and_exposure_chart(filled_df, output_path=None):
    """
    Create a simplified chart showing normalized equity and net exposure.
    Net exposure is calculated as long_exposure minus short_exposure
    and displayed as a bar chart without normalization.
    
    Args:
        filled_df (pandas.DataFrame): DataFrame with normalized equity and exposure values
        output_path (str, optional): Path to save the image file
    
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    # Make a copy of the dataframe to avoid modifying the original
    df = filled_df.copy()
    
    # Calculate net_exposure from long_exposure and short_exposure
    if 'long_exposure' in df.columns and 'short_exposure' in df.columns:
        df['net_exposure'] = df['long_exposure'] - df['short_exposure']
    elif 'long_exposure' in df.columns:
        df['net_exposure'] = df['long_exposure']
    elif 'short_exposure' in df.columns:
        df['net_exposure'] = -df['short_exposure']
    else:
        print("Warning: No exposure data found.")
        df['net_exposure'] = 0
    
    # Forward fill any NaN values in net_exposure
    df['net_exposure'] = df['net_exposure'].ffill().fillna(0)
    
    # Create figure with two y-axes
    fig, ax1 = plt.subplots(figsize=(12, 7))
    
    # Plot normalized equity on primary axis with primary color
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Normalized Equity', color=PRIMARY_COLOR)
    ax1.plot(df['date'], df['normalized_equity'], color=PRIMARY_COLOR, linewidth=2, label='Normalized Equity')
    
    # Create secondary Y axis for exposure
    ax2 = ax1.twinx()
    
    # Determine colors for bars based on positive or negative exposure
    positive_color = PRIMARY_COLOR  # Green for positive exposure
    negative_color = '#a83232'  # Red for negative exposure (keeping this for clarity)
    colors = [positive_color if x >= 0 else negative_color for x in df['net_exposure']]
    
    # Plot net exposure as bars on secondary axis
    ax2.bar(df['date'], df['net_exposure'], width=1, color=colors, alpha=0.6, label='Net Exposure')
    
    # Set y-axis label for exposure
    ax2.set_ylabel('Net Exposure (Long - Short)', color=PRIMARY_COLOR)
    
    # Add zero line for reference on exposure using secondary color
    ax2.axhline(y=0, color=SECONDARY_COLOR, linestyle='-', alpha=0.7)
    
    # Format x-axis for dates
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    plt.xticks(rotation=45)
    
    # Add title
    plt.title(f"Normalized Equity and Net Exposure - {df['account_name'].iloc[0]}")
    
    # Create combined legend for both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    # Add grid but only on the equity axis
    ax1.grid(True, alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if output path is provided
    if output_path:
        plt.savefig(output_path, dpi=150)
    
    return fig

def create_returns_distribution_chart(filled_df, output_path=None):
    """
    Create a chart plotting daily returns in ascending order as scatter dots.
    Includes vertical lines marking the 1st, 5th, and 10th percentiles.
    
    Args:
        filled_df (pandas.DataFrame): DataFrame with filled daily returns
        output_path (str, optional): Path to save the image file
    
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    # Filter out any NaN values
    returns = filled_df['filled_daily_return'].dropna()
    
    # Sort returns in ascending order
    sorted_returns = returns.sort_values()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create index for x-axis (percentile)
    x = np.linspace(0, 100, len(sorted_returns))
    
    # Plot returns as scatter dots with primary color
    ax.scatter(x, sorted_returns.values, color=PRIMARY_COLOR, s=15, alpha=0.7)
    
    # Add reference line at y=0 with secondary color
    ax.axhline(y=0, color=SECONDARY_COLOR, linestyle='-', alpha=0.7)
    
    # Add vertical lines at specific percentiles (1st, 5th, and 10th)
    percentiles_to_mark = [1, 5, 10]
    
    # Calculate the actual values at these percentiles
    for percentile in percentiles_to_mark:
        # Convert percentile to x-coordinate in the plot
        x_position = percentile
        
        # Find the corresponding y-value (return)
        index = int(percentile * len(sorted_returns) / 100)
        if index < len(sorted_returns):
            y_value = sorted_returns.iloc[index]
            
            # Add vertical line
            ax.axvline(x=x_position, color=SECONDARY_COLOR, linestyle='--', alpha=0.7)
            
            # Add annotation with the return value at this percentile
            label = f"{percentile}%: {y_value:.2%}"
            ax.text(x_position + 0.5, y_value, label, 
                    verticalalignment='center', fontsize=9,
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor=SECONDARY_COLOR))
    
    # Set labels and title
    ax.set_title(f"Daily Returns Distribution - {filled_df['account_name'].iloc[0]}")
    ax.set_xlabel('Percentile')
    ax.set_ylabel('Daily Return')
    
    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.2%}'.format(y)))
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if output path is provided
    if output_path:
        plt.savefig(output_path)
    
    return fig

def create_metrics_table(filled_df, output_path=None):
    """
    Create a table visualization of the performance metrics including total return.
    
    Args:
        filled_df (pandas.DataFrame): DataFrame with filled data
        output_path (str, optional): Path to save the image file
    
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    metrics = analyze_filled_returns(filled_df)
    
    # Format metrics for display
    formatted_metrics = {
        'Date Range': f"{filled_df['date'].min().strftime('%Y-%m-%d')} to {filled_df['date'].max().strftime('%Y-%m-%d')}",
        'Days Active': f"{metrics['days_active']} days",
        'Total Return': f"{metrics['total_return']:.2%}",
        'Annualized Return': f"{metrics['annualized_return']:.2%}",
        'Annualized Volatility': f"{metrics['annualized_volatility']:.2%}",
        'Sharpe Ratio': f"{metrics['sharpe_ratio']:.2f}",
        'Sortino Ratio': f"{metrics['sortino_ratio']:.2f}",
        'Maximum Drawdown': f"{metrics['max_drawdown']:.2%}",
        'Win Rate (Up Days)': f"{metrics['up_days']:.2%}",
        'Daily Average Return': f"{metrics['avg_daily_return']:.4%}",
        'Daily Volatility': f"{metrics['daily_volatility']:.4%}"
    }
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Hide axes
    ax.axis('tight')
    ax.axis('off')
    
    # Create table
    table_data = [[k, v] for k, v in formatted_metrics.items()]
    table = ax.table(cellText=table_data, colLabels=['Metric', 'Value'], 
                    loc='center', cellLoc='left', colWidths=[0.4, 0.4])
    
    # Style the table - using primary color for header
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.5)  # Adjust table size
    
    # Set header style using primary color
    for key, cell in table.get_celld().items():
        if key[0] == 0:  # Header row
            cell.set_facecolor(PRIMARY_COLOR)
            cell.set_text_props(color='white')
    
    # Set title
    ax.set_title(f"Performance Metrics", pad=20, fontsize=14)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if output path is provided
    if output_path:
        plt.savefig(output_path)
    
    return fig

def generate_pdf_report(filled_df):
    """
    Generate a complete PDF report with all the charts and tables.
    
    Args:
        account_df (pandas.DataFrame): Original account data
        normalized_df (pandas.DataFrame): Normalized equity data
        filled_df (pandas.DataFrame): Data with filled missing days
        metrics (dict): Performance metrics
        output_path (str): Path to save the PDF report
    
    Returns:
        str: Path to the saved PDF file
    """
    account_name = filled_df['account_name'].iloc[0]
    today = datetime.now().strftime('%Y-%m-%d')
    output_path = f"deep-analysis-2/{today}/{account_name}_{today}.pdf"

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Create PdfPages object to save multiple plots to single PDF
    with PdfPages(output_path) as pdf:
        # Add title page with primary color
        fig = plt.figure(figsize=(8.5, 11))
        fig.patch.set_facecolor('white')
        fig.text(0.5, 0.6, f"Performance Report", ha='center', fontsize=24, color=PRIMARY_COLOR)
        fig.text(0.5, 0.5, f"Account: {account_name}", ha='center', fontsize=18, color=PRIMARY_COLOR)
        fig.text(0.5, 0.4, f"Generated on: {datetime.now().strftime('%Y-%m-%d')}", ha='center', fontsize=14)
        pdf.savefig(fig)
        plt.close(fig)
        
        # Add equity curve chart
        fig = create_equity_curve_chart(filled_df)
        pdf.savefig(fig)
        plt.close(fig)
        
        # Add normalized equity chart
        fig = create_normalized_equity_chart(filled_df)
        pdf.savefig(fig)
        plt.close(fig)
        
        # Add equity and exposure chart
        fig = create_equity_and_exposure_chart(filled_df)
        pdf.savefig(fig)
        plt.close(fig)
        
        # Add returns distribution chart
        fig = create_returns_distribution_chart(filled_df)
        pdf.savefig(fig)
        plt.close(fig)
        
        # Add metrics table
        fig = create_metrics_table(filled_df)
        pdf.savefig(fig)
        plt.close(fig)
    
    print(f"PDF report generated successfully: {output_path}")
    return output_path

# COMBINED REPORT
def create_cash_flow_chart(df, output_path=None):
    """
    Create a chart showing total deposits and withdrawals over time.
    
    Args:
        df (pandas.DataFrame): DataFrame containing data for all accounts
        output_path (str, optional): Path to save the image file
    
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    # Make a copy and ensure data is sorted by date
    df_copy = df.copy()
    df_copy.sort_values('date', inplace=True)
    
    # Group by date and calculate total deposits and withdrawals
    cash_flow_df = df_copy.groupby('date').agg({
        'deposit': 'sum',
        'withdrawal': 'sum'
    }).reset_index()
    
    # Calculate cumulative deposits and withdrawals
    cash_flow_df['cumulative_deposits'] = cash_flow_df['deposit'].cumsum()
    cash_flow_df['cumulative_withdrawals'] = cash_flow_df['withdrawal'].cumsum()
    cash_flow_df['net_flow'] = cash_flow_df['cumulative_deposits'] - cash_flow_df['cumulative_withdrawals']
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot cumulative deposits with primary color
    ax.plot(cash_flow_df['date'], cash_flow_df['cumulative_deposits'], 
            color=PRIMARY_COLOR, linestyle='-', linewidth=2, label='Cumulative Deposits')
    
    # Plot cumulative withdrawals with red color
    withdrawal_color = '#a83232'  # Red for withdrawals
    ax.plot(cash_flow_df['date'], cash_flow_df['cumulative_withdrawals'], 
            color=withdrawal_color, linestyle='-', linewidth=2, label='Cumulative Withdrawals')
    
    # Plot net flow with secondary color
    ax.plot(cash_flow_df['date'], cash_flow_df['net_flow'], 
            color=SECONDARY_COLOR, linestyle='--', linewidth=2, label='Net Cash Flow')
    
    # Set title and labels
    ax.set_title("Cash Flow Analysis - All Accounts")
    ax.set_xlabel('Date')
    ax.set_ylabel('Amount ($)')
    
    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.xticks(rotation=45)
    
    # Add grid and legend
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if output path is provided
    if output_path:
        plt.savefig(output_path)
    
    return fig

def create_active_accounts_chart(df, output_path=None):
    """
    Create a chart showing the number of active accounts over time.
    An account is considered active if it has data on a given date.
    
    Args:
        df (pandas.DataFrame): DataFrame containing data for all accounts
        output_path (str, optional): Path to save the image file
    
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    # Make a copy and ensure data is sorted by date
    df_copy = df.copy()
    df_copy.sort_values('date', inplace=True)
    
    # Group by date and count unique accounts
    active_accounts_df = df_copy.groupby('date')['account_name'].nunique().reset_index()
    active_accounts_df.columns = ['date', 'active_accounts']
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot active accounts count with bars using primary color
    ax.bar(active_accounts_df['date'], active_accounts_df['active_accounts'], 
           color=PRIMARY_COLOR, alpha=0.7, width=1)
    
    # Plot line over bars for better visualization
    ax.plot(active_accounts_df['date'], active_accounts_df['active_accounts'], 
            color=PRIMARY_COLOR, linewidth=2)
    
    # Set title and labels
    ax.set_title("Active Accounts Over Time")
    ax.set_xlabel('Date')
    ax.set_ylabel('Number of Active Accounts')
    
    # Format y-axis to show integers
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    
    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.xticks(rotation=45)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if output path is provided
    if output_path:
        plt.savefig(output_path)
    
    return fig

def create_account_contribution_chart(df, output_path=None):
    """
    Create a chart showing the contribution of each account to the total AUM over time.
    
    Args:
        df (pandas.DataFrame): DataFrame containing data for all accounts
        output_path (str, optional): Path to save the image file
    
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    # Make a copy and ensure data is sorted by date
    df_copy = df.copy()
    df_copy.sort_values(['date', 'account_name'], inplace=True)
    
    # Get list of all dates and accounts
    all_dates = sorted(df_copy['date'].unique())
    all_accounts = sorted(df_copy['account_name'].unique())
    
    # Create a pivot table with dates as index, accounts as columns, and equity as values
    pivot_df = df_copy.pivot_table(
        index='date', 
        columns='account_name', 
        values='equity',
        aggfunc='sum'
    ).fillna(0)
    
    # Calculate percentage contribution of each account for each date
    for date in pivot_df.index:
        total = pivot_df.loc[date].sum()
        if total > 0:  # Avoid division by zero
            pivot_df.loc[date] = (pivot_df.loc[date] / total) * 100
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Create stacked area chart
    ax.stackplot(pivot_df.index, [pivot_df[account] for account in pivot_df.columns],
                 labels=pivot_df.columns, alpha=0.7)
    
    # Set title and labels
    ax.set_title("Account Contribution to Total AUM")
    ax.set_xlabel('Date')
    ax.set_ylabel('Contribution (%)')
    
    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.xticks(rotation=45)
    
    # Format y-axis as percentage
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0f}%'.format(y)))
    
    # Add legend with small font size
    ax.legend(loc='upper left', fontsize='small')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if output path is provided
    if output_path:
        plt.savefig(output_path)
    
    return fig

def create_aum_summary_table(df, output_path=None):
    """
    Create a table with summary statistics for the total AUM analysis.
    
    Args:
        df (pandas.DataFrame): DataFrame containing data for all accounts
        output_path (str, optional): Path to save the image file
    
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    # Make a copy
    df_copy = df.copy()
    
    # Calculate summary statistics
    total_accounts = df_copy['account_name'].nunique()
    date_range = f"{df_copy['date'].min().strftime('%Y-%m-%d')} to {df_copy['date'].max().strftime('%Y-%m-%d')}"
    days_of_data = (df_copy['date'].max() - df_copy['date'].min()).days
    
    # Total deposits and withdrawals
    total_deposits = df_copy['deposit'].sum()
    total_withdrawals = df_copy['withdrawal'].sum()
    net_cash_flow = total_deposits - total_withdrawals
    
    # Current AUM (most recent date)
    latest_date = df_copy['date'].max()
    current_aum = df_copy[df_copy['date'] == latest_date]['equity'].sum()
    
    # Format summary data for display
    summary_data = {
        'Total Unique Accounts': f"{total_accounts}",
        'Date Range': date_range,
        'Days of Data': f"{days_of_data}",
        'Total Deposits': f"${total_deposits:,.2f}",
        'Total Withdrawals': f"${total_withdrawals:,.2f}",
        'Net Cash Flow': f"${net_cash_flow:,.2f}",
        'Current AUM': f"${current_aum:,.2f}",
        'PnL': f"${current_aum - net_cash_flow:,.2f}"
    }
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Hide axes
    ax.axis('tight')
    ax.axis('off')
    
    # Create table
    table_data = [[k, v] for k, v in summary_data.items()]
    table = ax.table(cellText=table_data, colLabels=['Metric', 'Value'],
                    loc='center', cellLoc='left', colWidths=[0.4, 0.4])
    
    # Style the table - using primary color for header
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.5)  # Adjust table size
    
    # Set header style using primary color
    for key, cell in table.get_celld().items():
        if key[0] == 0:  # Header row
            cell.set_facecolor(PRIMARY_COLOR)
            cell.set_text_props(color='white')
    
    # Set title
    ax.set_title(f"AUM Summary Statistics", pad=20, fontsize=14)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if output path is provided
    if output_path:
        plt.savefig(output_path)
    
    return fig

def create_monthly_summary_chart(df, output_path=None):
    """
    Create a chart showing monthly deposits, withdrawals, and change in AUM.
    
    Args:
        df (pandas.DataFrame): DataFrame containing data for all accounts
        output_path (str, optional): Path to save the image file
    
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    # Make a copy
    df_copy = df.copy()
    
    # Convert date to datetime if it's not already
    df_copy['date'] = pd.to_datetime(df_copy['date'])
    
    # Extract year and month
    df_copy['year_month'] = df_copy['date'].dt.strftime('%Y-%m')
    
    # Group by year-month and calculate monthly metrics
    monthly_df = df_copy.groupby('year_month').agg({
        'deposit': 'sum',
        'withdrawal': 'sum',
        'date': 'max'  # Get the last day of each month for ordering
    }).reset_index()
    
    # Sort by date
    monthly_df.sort_values('date', inplace=True)
    
    # Calculate net flow
    monthly_df['net_flow'] = monthly_df['deposit'] - monthly_df['withdrawal']
    
    # Create figure with two y-axes
    fig, ax1 = plt.subplots(figsize=(12, 7))
    
    # Bar width
    width = 0.3
    
    # Positions for bars
    x = np.arange(len(monthly_df))
    
    # Plot deposits
    deposit_bars = ax1.bar(x - width/2, monthly_df['deposit'], width, label='Deposits', color=PRIMARY_COLOR, alpha=0.7)
    
    # Plot withdrawals as negative values
    withdrawal_bars = ax1.bar(x + width/2, -monthly_df['withdrawal'], width, label='Withdrawals', color='#a83232', alpha=0.7)
    
    # Plot net flow as a line on the same axis
    ax1.plot(x, monthly_df['net_flow'], marker='o', linestyle='-', color=SECONDARY_COLOR, linewidth=2, label='Net Flow')

    # Set labels for primary axis
    ax1.set_xlabel('Month')
    ax1.set_ylabel('Amount ($)')
    
    # Set x-tick labels as months
    ax1.set_xticks(x)
    ax1.set_xticklabels(monthly_df['year_month'], rotation=45)
    
    # Add title
    plt.title('Monthly Deposits and Withdrawals')
    
    # Add grid
    ax1.grid(True, alpha=0.3)
    
    # Add legend
    ax1.legend(loc='upper left')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if output path is provided
    if output_path:
        plt.savefig(output_path)
    
    return fig

def combined_report(df):
    """
    Calculate the total assets under management (sum of equity for all accounts) 
    for each day and generate a comprehensive report with additional analytics.
    
    Args:
        df (pandas.DataFrame): DataFrame containing data for all accounts
        
    Returns:
        str: Path to the saved PDF report
    """
    if df is None or df.empty:
        print("No data to calculate total AUM")
        return None
    
    # Make a copy to avoid modifying the original dataframe
    df_copy = df.copy()
    
    # Group by date and sum equity, deposits, and withdrawals
    total_df = df_copy.groupby('date').agg({
        'equity': 'sum',
        'deposit': 'sum',
        'withdrawal': 'sum',
        'long_exposure': 'sum',
        'short_exposure': 'sum'
    }).reset_index()
    
    # Add account_name for consistency with other functions
    total_df['account_name'] = 'total'
    
    # Process the data in the same way as individual accounts
    processed_aum = process_data(total_df)
    
    if processed_aum is not None:
        # Generate PDF report with a specific filename for the total AUM
        today = datetime.now().strftime('%Y-%m-%d')
        output_path = f"deep-analysis-2/{today}/total_{today}.pdf"
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Create PdfPages object to save multiple plots to single PDF
        with PdfPages(output_path) as pdf:
            # Add title page
            fig = plt.figure(figsize=(8.5, 11))
            fig.patch.set_facecolor('white')
            fig.text(0.5, 0.6, f"Report", ha='center', fontsize=24, color=PRIMARY_COLOR)
            fig.text(0.5, 0.5, f"All Accounts Combined", ha='center', fontsize=18, color=PRIMARY_COLOR)
            fig.text(0.5, 0.4, f"Generated on: {today}", ha='center', fontsize=14)
            pdf.savefig(fig)
            plt.close(fig)
            
            # Add AUM summary table (new)
            fig = create_aum_summary_table(df_copy)
            pdf.savefig(fig)
            plt.close(fig)
            
            # Add equity curve chart
            fig = create_equity_curve_chart(processed_aum)
            pdf.savefig(fig)
            plt.close(fig)
            
            # Add active accounts over time chart (new)
            fig = create_active_accounts_chart(df_copy)
            pdf.savefig(fig)
            plt.close(fig)
            
            # Add account contribution chart (new)
            fig = create_account_contribution_chart(df_copy)
            pdf.savefig(fig)
            plt.close(fig)
            
            # Add cash flow chart (new)
            fig = create_cash_flow_chart(total_df)
            pdf.savefig(fig)
            plt.close(fig)
            
            # Add monthly summary chart (new)
            fig = create_monthly_summary_chart(total_df)
            pdf.savefig(fig)
            plt.close(fig)
        
        print(f"PDF report generated successfully: {output_path}")
        return output_path
    
    return None

def main():
    db = "db/database.db"
    df = get_data(db)
    if df is None:
        print("No data retrieved. Exiting.")
        return

    # get active accounts from env
    accounts = [account['name'] for account in get_accounts_from_env()]
    # accounts = ['gabriele'] 
    
    # Calculate and report on total AUM across all accounts
    total_report = combined_report(df)

    # Process each individual account
    for account in accounts:
        account_df = get_account_data(df, account)
        processed_data = process_data(account_df)

        if processed_data is not None:
            report = generate_pdf_report(processed_data)


if __name__ == "__main__":
    main()