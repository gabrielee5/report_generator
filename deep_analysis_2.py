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
    Create a chart plotting daily returns in ascending order.
    
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
    
    # Plot returns with primary color
    ax.plot(x, sorted_returns.values, color=PRIMARY_COLOR, linestyle='-')
    
    # Add reference line at y=0 with secondary color
    ax.axhline(y=0, color=SECONDARY_COLOR, linestyle='-', alpha=0.7)
    
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

def main():
    db = "db/database.db"
    df = get_data(db)
    if df is None:
        print("No data retrieved. Exiting.")
        return

    # get active accounts from env
    accounts = [account['name'] for account in get_accounts_from_env()]
    # accounts = ['gabriele'] 

    for account in accounts:
        account_df = get_account_data(df, account)
        processed_data = process_data(account_df)

        if processed_data is not None:
            report = generate_pdf_report(processed_data)

if __name__ == "__main__":
    main()