import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import numpy as np
import os
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
import time

from main import get_accounts_from_env

class TradingAnalyzer:
    def __init__(self, db_path):
        """Initialize the analyzer with the database path."""
        self.db_path = db_path
        self.conn = None
        self.cursor = None
        
        # Create output directory for PDF reports
        today = datetime.now().strftime('%Y-%m-%d')
        self.output_dir = f"deep-analysis/{today}"
        os.makedirs(self.output_dir, exist_ok=True)
        
    def connect(self):
        """Connect to the SQLite database."""
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.cursor = self.conn.cursor()
            print(f"Successfully connected to {self.db_path}")
            return True
        except sqlite3.Error as e:
            print(f"Error connecting to database: {e}")
            return False
            
    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            print("Database connection closed.")
    
    def get_all_data(self):
        """Fetch all data from daily_reports table."""
        if not self.conn:
            if not self.connect():
                return None
        
        query = "SELECT * FROM daily_reports ORDER BY date"
        try:
            df = pd.read_sql_query(query, self.conn)
            # Convert date string to datetime object
            df['date'] = pd.to_datetime(df['date'])
            return df
        except sqlite3.Error as e:
            print(f"Error fetching data: {e}")
            return None
    
    def get_data_by_account(self, account_name):
        """Fetch data for a specific account."""
        if not self.conn:
            if not self.connect():
                return None
        
        query = "SELECT * FROM daily_reports WHERE account_name = ? ORDER BY date"
        try:
            df = pd.read_sql_query(query, self.conn, params=(account_name,))
            df['date'] = pd.to_datetime(df['date'])
            return df
        except sqlite3.Error as e:
            print(f"Error fetching data for account {account_name}: {e}")
            return None
    
    def get_data_by_date_range(self, start_date, end_date):
        """Fetch data within a specific date range."""
        if not self.conn:
            if not self.connect():
                return None
        
        query = "SELECT * FROM daily_reports WHERE date BETWEEN ? AND ? ORDER BY date"
        try:
            df = pd.read_sql_query(query, self.conn, params=(start_date, end_date))
            df['date'] = pd.to_datetime(df['date'])
            return df
        except sqlite3.Error as e:
            print(f"Error fetching data for date range {start_date} to {end_date}: {e}")
            return None
    
    def get_unique_accounts(self):
        """Get a list of all unique account names."""
        if not self.conn:
            if not self.connect():
                return None
        
        query = "SELECT DISTINCT account_name FROM daily_reports"
        try:
            result = self.cursor.execute(query).fetchall()
            return [r[0] for r in result]
        except sqlite3.Error as e:
            print(f"Error fetching unique accounts: {e}")
            return None
    
    def fill_missing_days(self, df=None, account_name=None):
        """Fill missing days in the dataset with interpolated values from the day before and after.
        
        Parameters:
        -----------
        df : pandas.DataFrame, optional
            The dataframe to process. If None, data will be fetched based on account_name.
        account_name : str, optional
            The account name to fetch data for if df is None.
            
        Returns:
        --------
        pandas.DataFrame
            The dataframe with filled missing days.
        """
        # Get data if not provided
        if df is None:
            if account_name:
                df = self.get_data_by_account(account_name)
            else:
                df = self.get_all_data()
                
        if df is None or df.empty:
            print("No data available for filling missing days")
            return None
        
        # Make a copy to avoid modifying the original dataframe
        df_filled = df.copy()
        
        # Process each account separately
        filled_dfs = []
        
        for name, group in df_filled.groupby('account_name'):
            # Sort by date
            group = group.sort_values('date')
            
            # Create a complete date range
            date_range = pd.date_range(start=group['date'].min(), end=group['date'].max(), freq='D')
            
            # Create a new dataframe with the complete date range
            temp_df = pd.DataFrame({'date': date_range})
            
            # Merge with the existing data to identify missing days
            merged_df = pd.merge(temp_df, group, on='date', how='left')
            
            # Fill account_name for missing days
            merged_df['account_name'] = merged_df['account_name'].fillna(name)
            
            # Find indices of missing days (rows with NaN values)
            missing_indices = merged_df[merged_df['equity'].isna()].index
            
            for idx in missing_indices:
                # Find the indices of the previous and next available data points
                prev_idx = merged_df.iloc[:idx][~merged_df.iloc[:idx]['equity'].isna()].index.max() if idx > 0 else None
                next_idx = merged_df.iloc[idx+1:][~merged_df.iloc[idx+1:]['equity'].isna()].index.min() if idx < len(merged_df) - 1 else None
                
                # Skip if we can't find both previous and next data points
                if prev_idx is None or next_idx is None:
                    continue
                
                # Get the values from previous and next days
                prev_values = merged_df.loc[prev_idx]
                next_values = merged_df.loc[next_idx]
                
                # Calculate the number of days between data points
                days_diff = (next_idx - prev_idx) + 1
                
                # Calculate linear interpolation weights
                weight_next = (idx - prev_idx) / (days_diff - 1)
                weight_prev = 1 - weight_next
                
                # Interpolate numeric columns (focusing on equity and exposure)
                for col in ['equity', 'long_exposure', 'short_exposure']:
                    if col in merged_df.columns:
                        merged_df.loc[idx, col] = weight_prev * prev_values[col] + weight_next * next_values[col]
                
                # For other numeric columns, use the same interpolation method
                for col in ['open_positions', 'long_positions', 'short_positions']:
                    if col in merged_df.columns:
                        # Round to nearest integer for count-based columns
                        merged_df.loc[idx, col] = round(weight_prev * prev_values[col] + weight_next * next_values[col])
                
                # For the remaining columns, use forward fill from previous day
                for col in merged_df.columns:
                    if col not in ['date', 'account_name', 'equity', 'long_exposure', 'short_exposure', 
                                'open_positions', 'long_positions', 'short_positions'] and col in merged_df.columns:
                        merged_df.loc[idx, col] = prev_values[col]
                
                # Set trades_today, funding_fees, trading_fees, total_volume, deposit, withdrawal to 0
                # These are daily metrics that shouldn't be interpolated
                for col in ['trades_today', 'funding_fees', 'trading_fees', 'total_volume', 'deposit', 'withdrawal']:
                    if col in merged_df.columns:
                        merged_df.loc[idx, col] = 0
            
            filled_dfs.append(merged_df)
        
        # Combine all processed dataframes
        result_df = pd.concat(filled_dfs)
        
        return result_df

    def calculate_basic_metrics(self, df=None):
        """Calculate basic metrics from the data."""
        if df is None:
            df = self.get_all_data()
            if df is None:
                return None
        
        # Group by account and calculate metrics
        account_metrics = df.groupby('account_name').agg({
            'equity': ['mean', 'min', 'max', 'std'],
            'open_positions': 'mean',
            'trades_today': 'sum',
            'funding_fees': 'sum',
            'trading_fees': 'sum',
            'total_volume': 'sum',
            'deposit': 'sum',
            'withdrawal': 'sum'
        })
        
        # Calculate net deposits
        account_metrics['net_deposits'] = account_metrics[('deposit', 'sum')] - account_metrics[('withdrawal', 'sum')]
        
        # Calculate total fees
        account_metrics['total_fees'] = account_metrics[('funding_fees', 'sum')] + account_metrics[('trading_fees', 'sum')]
        
        # Calculate fee as percentage of volume
        account_metrics['fee_percentage'] = (account_metrics['total_fees'] / account_metrics[('total_volume', 'sum')]) * 100
        
        return account_metrics
    
    def plot_equity_over_time(self, account_name=None):
        """Plot equity over time for all accounts or a specific account."""
        if account_name:
            df = self.get_data_by_account(account_name)
            title = f"Equity Over Time for {account_name}"
        else:
            df = self.get_all_data()
            title = "Equity Over Time for All Accounts"
        
        if df is None or df.empty:
            print("No data available for plotting")
            return
        
        plt.figure(figsize=(12, 6))
        
        if account_name:
            plt.plot(df['date'], df['equity'])
        else:
            for name, group in df.groupby('account_name'):
                plt.plot(group['date'], group['equity'], label=name)
            plt.legend()
        
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Equity')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    def plot_daily_pnl(self, account_name=None):
        """Plot daily PnL based on equity changes."""
        if account_name:
            df = self.get_data_by_account(account_name)
            title = f"Daily PnL for {account_name}"
        else:
            df = self.get_all_data()
            title = "Daily PnL for All Accounts"
        
        if df is None or df.empty:
            print("No data available for plotting")
            return
        
        # Calculate daily PnL
        df = df.sort_values(['account_name', 'date'])
        df['prev_equity'] = df.groupby('account_name')['equity'].shift(1)
        df['daily_deposits'] = df['deposit'] - df['withdrawal']
        df['daily_pnl'] = df['equity'] - df['prev_equity'] - df['daily_deposits']
        
        plt.figure(figsize=(12, 6))
        
        if account_name:
            valid_data = df.dropna(subset=['daily_pnl'])
            plt.bar(valid_data['date'], valid_data['daily_pnl'])
        else:
            for name, group in df.groupby('account_name'):
                valid_data = group.dropna(subset=['daily_pnl'])
                plt.plot(valid_data['date'], valid_data['daily_pnl'], label=name)
            plt.legend()
        
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Daily PnL')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    def analyze_exposure_balance(self, account_name=None):
        """Analyze the balance between long and short exposure."""
        if account_name:
            df = self.get_data_by_account(account_name)
            title = f"Exposure Balance for {account_name}"
        else:
            df = self.get_all_data()
            title = "Exposure Balance for All Accounts"
        
        if df is None or df.empty:
            print("No data available for analysis")
            return
        
        df['net_exposure'] = df['long_exposure'] - df['short_exposure'].abs()
        df['exposure_ratio'] = df['long_exposure'] / df['short_exposure'].abs()
        df['total_exposure'] = df['long_exposure'] + df['short_exposure'].abs()
        df['exposure_bias'] = df['net_exposure'] / df['total_exposure']
        
        result = df.groupby('account_name').agg({
            'net_exposure': 'mean',
            'exposure_ratio': 'mean',
            'exposure_bias': 'mean',
            'long_positions': 'mean',
            'short_positions': 'mean'
        })
        
        return result
    
    def calculate_normalized_equity(self, df):
        """Calculate normalized equity curve by computing daily returns adjusted for deposits/withdrawals 
        and showing the growth of an initial 100 investment."""
        if df is None or df.empty:
            return df
            
        # Sort data by date
        df = df.sort_values('date').reset_index(drop=True)
        
        # Calculate daily raw change in equity
        df['prev_equity'] = df['equity'].shift(1)
        df['equity_change'] = df['equity'] - df['prev_equity']
        
        # Adjust the change for deposits/withdrawals
        df['adjusted_change'] = df['equity_change'] - df['deposit'] + df['withdrawal']
        
        # Calculate daily return rate (percentage change)
        df['daily_return'] = df['adjusted_change'] / df['prev_equity'].replace(0, np.nan)
        
        # Start normalized equity at 100
        df['normalized_equity'] = 100.0
        
        # Calculate cumulative performance starting at 100
        for i in range(1, len(df)):
            if not np.isnan(df.loc[i, 'daily_return']):
                df.loc[i, 'normalized_equity'] = df.loc[i-1, 'normalized_equity'] * (1 + df.loc[i, 'daily_return'])
            else:
                df.loc[i, 'normalized_equity'] = df.loc[i-1, 'normalized_equity']
        
        return df
    
    def calculate_performance_metrics(self, account_name=None):
        """Calculate performance metrics including Sharpe and Sortino ratios based on normalized equity."""
        if account_name:
            df = self.get_data_by_account(account_name)
        else:
            df = self.get_all_data()
        
        if df is None or df.empty:
            print("No data available for analysis")
            return
        
        # Fill missing days before continuing with calculations
        df = self.fill_missing_days(df)

        # Calculate metrics by account
        metrics = {}
        risk_free_rate = 0.02 / 365  # Assume 2% annual risk-free rate, daily
        
        for name, group in df.groupby('account_name'):
            # Calculate normalized equity and returns
            norm_df = self.calculate_normalized_equity(group.copy())
            
            valid_data = norm_df.dropna(subset=['daily_return'])
            if len(valid_data) < 2:
                continue
                
            # Performance metrics
            returns = valid_data['daily_return']
            avg_return = returns.mean()
            std_return = returns.std()
            
            # Sharpe Ratio (annualized)
            excess_return = avg_return - risk_free_rate
            sharpe = excess_return / std_return * np.sqrt(365) if std_return > 0 else 0
            
            # Sortino Ratio
            downside_returns = returns[returns < 0]
            downside_deviation = downside_returns.std() if len(downside_returns) > 0 else 0
            sortino = excess_return / downside_deviation * np.sqrt(365) if downside_deviation > 0 else 0
            
            # Win rate (positive return days)
            win_rate = (returns > 0).mean()
            
            # Maximum drawdown
            cumulative_returns = (1 + returns).cumprod()
            running_max = cumulative_returns.cummax()
            drawdown = (cumulative_returns / running_max) - 1
            max_drawdown = drawdown.min()
            
            # Calmar Ratio (annualized return / max drawdown)
            annualized_return = (1 + avg_return) ** 365 - 1
            calmar = abs(annualized_return / max_drawdown) if max_drawdown < 0 else 0
            
            metrics[name] = {
                'avg_daily_return': avg_return,
                'annualized_return': annualized_return,
                'daily_volatility': std_return,
                'annualized_volatility': std_return * np.sqrt(365),
                'sharpe_ratio': sharpe,
                'sortino_ratio': sortino,
                'calmar_ratio': calmar,
                'win_rate': win_rate,
                'max_drawdown': max_drawdown
            }
        
        return pd.DataFrame(metrics).T
    
    def analyze_fee_impact(self, account_name=None):
        """Analyze the impact of fees on performance."""
        if account_name:
            df = self.get_data_by_account(account_name)
        else:
            df = self.get_all_data()
        
        if df is None or df.empty:
            print("No data available for analysis")
            return
        
        # Group by account
        fee_analysis = df.groupby('account_name').agg({
            'funding_fees': 'sum',
            'trading_fees': 'sum',
            'total_volume': 'sum',
            'equity': 'last'
        })
        
        fee_analysis['total_fees'] = fee_analysis['funding_fees'] + fee_analysis['trading_fees']
        fee_analysis['fee_percentage_of_volume'] = fee_analysis['total_fees'] / fee_analysis['total_volume'] * 100
        fee_analysis['fee_percentage_of_equity'] = fee_analysis['total_fees'] / fee_analysis['equity'] * 100
        
        return fee_analysis
    
    def generate_report(self, account_name=None):
        """Generate a comprehensive report of all metrics."""
        if account_name:
            print(f"=== Trading Analysis Report for {account_name} ===")
        else:
            print("=== Trading Analysis Report for All Accounts ===")
        
        # Get basic metrics
        basic_metrics = self.calculate_basic_metrics()
        if basic_metrics is not None:
            print("\n--- Basic Metrics ---")
            print(basic_metrics)
        
        # Get exposure balance
        exposure_balance = self.analyze_exposure_balance()
        if exposure_balance is not None:
            print("\n--- Exposure Balance ---")
            print(exposure_balance)
        
        # Get performance metrics
        performance_metrics = self.calculate_performance_metrics()
        if performance_metrics is not None:
            print("\n--- Performance Metrics ---")
            print(performance_metrics)
        
        # Get fee impact
        fee_impact = self.analyze_fee_impact()
        if fee_impact is not None:
            print("\n--- Fee Impact Analysis ---")
            print(fee_impact)


    def generate_pdf_report(self, account_name=None):
        """Generate a comprehensive PDF report for an account or all accounts."""
        if account_name:
            accounts = [account_name]
        else:
            accounts = self.get_unique_accounts() # fall back on this
            
        if not accounts:
            print("No accounts found for analysis")
            return
            
        today = datetime.now().strftime('%Y-%m-%d')
        
        for acc in accounts:
            # Create PDF file
            pdf_path = f"{self.output_dir}/{acc}_{today}.pdf"
            with PdfPages(pdf_path) as pdf:
                # Get data for this account
                df = self.get_data_by_account(acc)
                if df is None or df.empty:
                    print(f"No data available for account {acc}")
                    continue
                
                # Calculate normalized equity
                norm_df = self.calculate_normalized_equity(df.copy())
                
                # 1. Title page
                plt.figure(figsize=(8.5, 11))
                plt.axis('off')
                plt.text(0.5, 0.5, f"Trading Performance Analysis\nAccount: {acc}\nDate: {today}",
                        horizontalalignment='center', verticalalignment='center', fontsize=20)
                pdf.savefig()
                plt.close()
                
                # 2. Create figure with separate graphs for equity curves
                fig = plt.figure(figsize=(10, 15))
                gs = gridspec.GridSpec(4, 1, height_ratios=[2, 2, 1, 1])
                
                # Plot raw equity curve
                ax1 = plt.subplot(gs[0])
                ax1.plot(df['date'], df['equity'], label='Raw Equity', color='green', linewidth=2)
                ax1.set_title(f'Raw Equity Curve for {acc}', fontsize=14)
                ax1.set_ylabel('Equity Value', fontsize=12)
                ax1.legend()
                ax1.grid(True)
                
                # Plot normalized equity curve
                ax2 = plt.subplot(gs[1])
                ax2.plot(norm_df['date'], norm_df['normalized_equity'], label='Normalized Equity (Starting at 100)', color='blue', linewidth=2)
                ax2.set_title(f'Normalized Equity Curve for {acc} (Starting at 100)', fontsize=14)
                ax2.set_ylabel('Normalized Equity Value', fontsize=12)
                ax2.legend()
                ax2.grid(True)
                
                # Plot deposits and withdrawals
                ax3 = plt.subplot(gs[2], sharex=ax1)
                ax3.bar(df['date'], df['deposit'], label='Deposits', color='green', alpha=0.7)
                ax3.bar(df['date'], -df['withdrawal'], label='Withdrawals', color='red', alpha=0.7)
                ax3.set_title('Deposits and Withdrawals', fontsize=14)
                ax3.set_ylabel('Amount', fontsize=12)
                ax3.legend()
                ax3.grid(True)
                
                # Plot net exposure and normalized equity
                ax4 = plt.subplot(gs[3], sharex=ax1)
                ax4.bar(df['date'], df['long_exposure'] - df['short_exposure'].abs(), label='Net Exposure', color='purple', alpha=0.7)
                
                # Create a twin y-axis for the normalized equity
                ax4_twin = ax4.twinx()
                ax4_twin.plot(norm_df['date'], norm_df['normalized_equity'], label='Normalized Equity (100 Base)', color='blue', linestyle='--')
                
                ax4.set_title('Net Exposure vs Normalized Equity', fontsize=14)
                ax4.set_xlabel('Date', fontsize=12)
                ax4.set_ylabel('Net Exposure', fontsize=12)
                ax4_twin.set_ylabel('Normalized Equity (Starting at 100)', fontsize=12)
                
                # Combine legends from both axes
                lines1, labels1 = ax4.get_legend_handles_labels()
                lines2, labels2 = ax4_twin.get_legend_handles_labels()
                ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
                
                ax3.grid(True)
                plt.tight_layout()
                pdf.savefig(fig)
                plt.close()
                
                # 3. Performance Metrics
                metrics = self.calculate_performance_metrics(acc)
                if metrics is not None and not metrics.empty:
                    plt.figure(figsize=(10, 8))
                    plt.axis('off')
                    
                    metrics_text = "Performance Metrics:\n\n"
                    metrics = metrics.iloc[0]  # Get the first row since we're analyzing a single account
                    
                    metrics_text += f"Annualized Return: {metrics['annualized_return']:.2%}\n"
                    metrics_text += f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}\n"
                    metrics_text += f"Sortino Ratio: {metrics['sortino_ratio']:.2f}\n"
                    metrics_text += f"Calmar Ratio: {metrics['calmar_ratio']:.2f}\n"
                    metrics_text += f"Win Rate: {metrics['win_rate']:.2%}\n"
                    metrics_text += f"Maximum Drawdown: {metrics['max_drawdown']:.2%}\n"
                    metrics_text += f"Annualized Volatility: {metrics['annualized_volatility']:.2%}\n"
                    
                    plt.text(0.1, 0.9, metrics_text, fontsize=12, va='top', family='monospace')
                    pdf.savefig()
                    plt.close()
                
                # 4. Drawdown chart
                valid_data = norm_df.dropna(subset=['daily_return'])
                if len(valid_data) >= 2:
                    cumulative_returns = (1 + valid_data['daily_return']).cumprod()
                    running_max = cumulative_returns.cummax()
                    drawdown = (cumulative_returns / running_max - 1) * 100  # Convert to percentage
                    
                    plt.figure(figsize=(10, 6))
                    plt.plot(valid_data['date'], drawdown, color='red', linewidth=1.5)
                    plt.fill_between(valid_data['date'], drawdown, 0, color='red', alpha=0.3)
                    plt.title('Drawdown Chart', fontsize=14)
                    plt.ylabel('Drawdown (%)', fontsize=12)
                    plt.xlabel('Date', fontsize=12)
                    plt.grid(True)
                    
                    # Format y-axis as percentage
                    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.0f}%'))
                    
                    plt.tight_layout()
                    pdf.savefig()
                    plt.close()
                
                # 5. Monthly returns heatmap
                if len(valid_data) >= 30:
                    # Make a copy to avoid the SettingWithCopyWarning
                    monthly_data = valid_data.copy()
                    monthly_data.loc[:, 'year'] = monthly_data['date'].dt.year
                    monthly_data.loc[:, 'month'] = monthly_data['date'].dt.month
                    
                    # Calculate monthly returns
                    monthly_returns = monthly_data.groupby(['year', 'month'])['daily_return'].apply(
                        lambda x: (1 + x).prod() - 1
                    ).unstack()
                    
                    # Create a heatmap
                    plt.figure(figsize=(10, 6))
                    sns.heatmap(monthly_returns * 100, annot=True, fmt=".2f", cmap="RdYlGn", 
                                cbar_kws={'label': 'Monthly Return (%)'})
                    plt.title('Monthly Returns Heatmap (%)', fontsize=14)
                    plt.xlabel('Month', fontsize=12)
                    plt.ylabel('Year', fontsize=12)
                    plt.tight_layout()
                    pdf.savefig()
                    plt.close()
                
                # 6. Long/Short position analysis
                plt.figure(figsize=(10, 6))
                plt.bar(df['date'], df['long_positions'], label='Long Positions', color='green', alpha=0.6)
                plt.bar(df['date'], -df['short_positions'], label='Short Positions', color='red', alpha=0.6)
                plt.title('Long vs Short Positions', fontsize=14)
                plt.xlabel('Date', fontsize=12)
                plt.ylabel('Number of Positions', fontsize=12)
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                pdf.savefig()
                plt.close()
                
                # 7. Fee analysis
                plt.figure(figsize=(10, 6))
                width = 0.35
                plt.bar(df['date'], df['trading_fees'], width, label='Trading Fees', color='orange')
                plt.bar(df['date'], df['funding_fees'], width, bottom=df['trading_fees'], label='Funding Fees', color='purple')
                plt.title('Fee Analysis', fontsize=14)
                plt.xlabel('Date', fontsize=12)
                plt.ylabel('Fees', fontsize=12)
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                pdf.savefig()
                plt.close()
                
                # 8. Trading volume
                plt.figure(figsize=(10, 6))
                plt.bar(df['date'], df['total_volume'], color='blue', alpha=0.7)
                plt.title('Trading Volume', fontsize=14)
                plt.xlabel('Date', fontsize=12)
                plt.ylabel('Volume', fontsize=12)
                plt.grid(True)
                plt.tight_layout()
                pdf.savefig()
                plt.close()
            
            print(f"PDF report for {acc} generated successfully: {pdf_path}")
        
        return True


if __name__ == "__main__":
    # Replace with your actual database path
    analyzer = TradingAnalyzer("db/database.db")
    
    # Connect to the database
    if analyzer.connect():
        # Get all unique accounts
        accounts = [account['name'] for account in get_accounts_from_env()]
        
        # Generate PDF reports for each account
        for account in accounts:
            analyzer.generate_pdf_report(account)
        
        # Or generate reports for all accounts at once
        # analyzer.generate_pdf_report()
        
        # Close the connection
        analyzer.close()