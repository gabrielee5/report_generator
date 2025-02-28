#!/usr/bin/env python3
# trading_analysis.py
# Deep analysis of trading data from database

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime, timedelta
import os
from scipy import stats
import warnings
import argparse
from decimal import Decimal
import matplotlib.ticker as ticker

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('ggplot')
sns.set_theme(style="whitegrid")

# Change to the directory where the script is located
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Database path
DB_PATH = 'db/database.db'
REPORTS_DIR = 'analysis_reports'

# Create reports directory if it doesn't exist
os.makedirs(REPORTS_DIR, exist_ok=True)

class TradingAnalyzer:
    """Class for analyzing trading data from SQLite database"""
    
    def __init__(self, db_path=DB_PATH):
        """Initialize analyzer with database path"""
        self.db_path = db_path
        self.accounts = None
        self.data = None
        self.daily_data = None
        self.weekly_data = None
        self.monthly_data = None
        
    def connect_db(self):
        """Create a connection to the SQLite database"""
        try:
            conn = sqlite3.connect(self.db_path)
            return conn
        except sqlite3.Error as e:
            print(f"Error connecting to database: {e}")
            raise
    
    def get_accounts(self):
        """Get list of all accounts in the database"""
        conn = self.connect_db()
        try:
            query = "SELECT DISTINCT account_name FROM daily_reports"
            self.accounts = pd.read_sql_query(query, conn)['account_name'].tolist()
            return self.accounts
        except sqlite3.Error as e:
            print(f"Error getting accounts: {e}")
            raise
        finally:
            conn.close()
    
    def load_data(self, days=None, account=None):
        """
        Load data from database
        
        Args:
            days (int, optional): Number of days to look back
            account (str, optional): Specific account to analyze
        """
        conn = self.connect_db()
        try:
            # Build the query
            query = "SELECT * FROM daily_reports"
            conditions = []
            
            if days:
                cutoff_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
                conditions.append(f"date >= '{cutoff_date}'")
                
            if account:
                conditions.append(f"account_name = '{account}'")
                
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
                
            query += " ORDER BY date ASC"
            
            # Load the data
            df = pd.read_sql_query(query, conn)
            
            # Convert date to datetime
            df['date'] = pd.to_datetime(df['date'])
            
            # Sort by date
            df = df.sort_values('date')
            
            self.data = df
            return df
            
        except sqlite3.Error as e:
            print(f"Error loading data: {e}")
            raise
        finally:
            conn.close()
    
    def calculate_returns(self, account=None):
        """
        Calculate daily returns accounting for deposits and withdrawals
        
        Args:
            account (str, optional): Specific account to analyze
        """
        # Make sure data is loaded
        if self.data is None:
            self.load_data(account=account)
            
        df = self.data.copy()
        
        if account:
            df = df[df['account_name'] == account]
            
        # Calculate daily returns for each account
        all_returns = []
        
        for account_name in df['account_name'].unique():
            account_df = df[df['account_name'] == account_name].copy()
            account_df = account_df.sort_values('date')
            
            # Calculate adjusted daily returns
            account_df['prev_equity'] = account_df['equity'].shift(1)
            account_df['deposit_adj'] = account_df['deposit'].fillna(0)
            account_df['withdrawal_adj'] = account_df['withdrawal'].fillna(0)
            account_df['adj_equity'] = account_df['equity'] - account_df['deposit_adj'] + account_df['withdrawal_adj']
            
            # Calculate return (ignore first row where prev_equity is NaN)
            account_df['daily_return'] = np.where(
                account_df['prev_equity'] > 0,
                (account_df['adj_equity'] - account_df['prev_equity']) / account_df['prev_equity'],
                np.nan
            )
            
            # Add account name
            account_df['account_name'] = account_name
            all_returns.append(account_df)
            
        # Combine all returns
        returns_df = pd.concat(all_returns)
        returns_df = returns_df[['date', 'account_name', 'equity', 'daily_return', 'deposit_adj', 'withdrawal_adj']]
        
        return returns_df
    
    def calculate_metrics(self, returns_df=None, account=None):
        """
        Calculate various trading metrics
        
        Args:
            returns_df (DataFrame, optional): DataFrame with daily returns
            account (str, optional): Specific account to analyze
        """
        if returns_df is None:
            returns_df = self.calculate_returns(account=account)
            
        metrics = {}
        
        for account_name in returns_df['account_name'].unique():
            account_returns = returns_df[returns_df['account_name'] == account_name]
            
            # Daily metrics
            daily_returns = account_returns['daily_return'].dropna()
            
            # Skip if not enough data
            if len(daily_returns) < 3:
                continue
                
            # Calculate metrics
            metrics[account_name] = {
                'total_days': len(daily_returns),
                'total_return': (1 + daily_returns).prod() - 1,
                'annualized_return': (1 + daily_returns).prod() ** (252 / len(daily_returns)) - 1,
                'volatility': daily_returns.std() * np.sqrt(252),
                'sharpe_ratio': (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() > 0 else 0,
                'max_drawdown': (account_returns['equity'] / account_returns['equity'].cummax() - 1).min(),
                'win_rate': (daily_returns > 0).mean(),
                'best_day': daily_returns.max(),
                'worst_day': daily_returns.min(),
                'avg_gain': daily_returns[daily_returns > 0].mean() if len(daily_returns[daily_returns > 0]) > 0 else 0,
                'avg_loss': daily_returns[daily_returns < 0].mean() if len(daily_returns[daily_returns < 0]) > 0 else 0,
                'profit_factor': abs(daily_returns[daily_returns > 0].sum() / daily_returns[daily_returns < 0].sum()) if daily_returns[daily_returns < 0].sum() != 0 else float('inf'),
                'last_equity': account_returns['equity'].iloc[-1]
            }
            
        return metrics
    
    def create_equity_curve(self, account=None, days=None, figsize=(12, 6)):
        """
        Create equity curve for specified account(s)
        
        Args:
            account (str, optional): Specific account to analyze
            days (int, optional): Number of days to look back
            figsize (tuple, optional): Figure size
        """
        if self.data is None:
            self.load_data(days=days, account=account)
            
        df = self.data.copy()
        
        if account:
            df = df[df['account_name'] == account]
            
        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)
        
        for account_name in df['account_name'].unique():
            account_df = df[df['account_name'] == account_name]
            ax.plot(account_df['date'], account_df['equity'], label=account_name)
            
        # Format the plot
        ax.set_title('Equity Curve' + (f' - {account}' if account else ''))
        ax.set_xlabel('Date')
        ax.set_ylabel('Equity (USDT)')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.xticks(rotation=45)
        
        # Format y-axis to show thousands with K
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x/1000:.1f}K'))
        
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return fig
    
    def create_drawdown_chart(self, account=None, days=None, figsize=(12, 6)):
        """
        Create drawdown chart for specified account(s)
        
        Args:
            account (str, optional): Specific account to analyze
            days (int, optional): Number of days to look back
            figsize (tuple, optional): Figure size
        """
        if self.data is None:
            self.load_data(days=days, account=account)
            
        df = self.data.copy()
        
        if account:
            df = df[df['account_name'] == account]
            
        if df.empty:
            # Create an empty figure with a message if no data
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, 'No data available for drawdown analysis', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes)
            plt.tight_layout()
            return fig
            
        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)
        
        for account_name in df['account_name'].unique():
            account_df = df[df['account_name'] == account_name]
            
            # Skip if only one data point
            if len(account_df) <= 1:
                continue
                
            # Calculate running maximum
            account_df['running_max'] = account_df['equity'].cummax()
            
            # Calculate drawdown percentage
            account_df['drawdown'] = (account_df['equity'] / account_df['running_max'] - 1) * 100
            
            ax.plot(account_df['date'], account_df['drawdown'], label=account_name)
            
        # Format the plot
        ax.set_title('Drawdown Analysis' + (f' - {account}' if account else ''))
        ax.set_xlabel('Date')
        ax.set_ylabel('Drawdown (%)')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.xticks(rotation=45)
        
        # Add zero line
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        
        # Check if we have data to determine y-axis limits
        if not any(account_df['equity'].cummax().max() for account_name in df['account_name'].unique() 
                 for account_df in [df[df['account_name'] == account_name]] if len(account_df) > 1):
            # No valid data, set default limits
            ax.set_ylim(bottom=-10, top=5)
        else:
            # Calculate the minimum drawdown across all accounts
            min_drawdown = min(
                (account_df['equity'].min() / account_df['equity'].cummax().max() * 100 - 5)
                for account_name in df['account_name'].unique()
                for account_df in [df[df['account_name'] == account_name]]
                if len(account_df) > 1 and account_df['equity'].cummax().max() > 0
            )
            ax.set_ylim(bottom=min_drawdown, top=5)
        
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return fig
    
    def analyze_exposure(self, account=None, days=None, figsize=(12, 6)):
        """
        Analyze long vs short exposure over time
        
        Args:
            account (str, optional): Specific account to analyze
            days (int, optional): Number of days to look back
            figsize (tuple, optional): Figure size
        """
        if self.data is None:
            self.load_data(days=days, account=account)
            
        df = self.data.copy()
        
        if account:
            df = df[df['account_name'] == account]
            
        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)
        
        for account_name in df['account_name'].unique():
            account_df = df[df['account_name'] == account_name]
            
            # Calculate net exposure
            account_df['net_exposure'] = account_df['long_exposure'] - account_df['short_exposure']
            account_df['net_exposure_pct'] = account_df['net_exposure'] / account_df['equity'] * 100
            
            ax.plot(account_df['date'], account_df['net_exposure_pct'], label=account_name)
            
        # Format the plot
        ax.set_title('Net Exposure (% of Equity)' + (f' - {account}' if account else ''))
        ax.set_xlabel('Date')
        ax.set_ylabel('Net Exposure (%)')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.xticks(rotation=45)
        
        # Add zero line for market neutral
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return fig
    
    def analyze_trading_activity(self, account=None, days=None, figsize=(12, 6)):
        """
        Analyze trading activity (number of trades, volume)
        
        Args:
            account (str, optional): Specific account to analyze
            days (int, optional): Number of days to look back
            figsize (tuple, optional): Figure size
        """
        if self.data is None:
            self.load_data(days=days, account=account)
            
        df = self.data.copy()
        
        if account:
            df = df[df['account_name'] == account]
            
        # Create the plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
        
        for account_name in df['account_name'].unique():
            account_df = df[df['account_name'] == account_name]
            
            # Plot number of trades
            ax1.plot(account_df['date'], account_df['trades_today'], label=account_name)
            
            # Plot trading volume
            ax2.plot(account_df['date'], account_df['total_volume'], label=account_name)
            
        # Format the plots
        ax1.set_title('Trading Activity' + (f' - {account}' if account else ''))
        ax1.set_ylabel('Number of Trades')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Trading Volume (USDT)')
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.xticks(rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Format y-axis to show thousands with K for volume
        ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x/1000:.1f}K'))
        
        plt.tight_layout()
        
        return fig
    
    def analyze_fees(self, account=None, days=None, figsize=(12, 6)):
        """
        Analyze trading and funding fees over time
        
        Args:
            account (str, optional): Specific account to analyze
            days (int, optional): Number of days to look back
            figsize (tuple, optional): Figure size
        """
        if self.data is None:
            self.load_data(days=days, account=account)
            
        df = self.data.copy()
        
        if account:
            df = df[df['account_name'] == account]
            
        # Create the plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
        
        for account_name in df['account_name'].unique():
            account_df = df[df['account_name'] == account_name]
            
            # Plot trading fees
            ax1.plot(account_df['date'], account_df['trading_fees'], label=f"{account_name} - Trading")
            
            # Plot funding fees
            ax2.plot(account_df['date'], account_df['funding_fees'], label=f"{account_name} - Funding")
            
        # Format the plots
        ax1.set_title('Fee Analysis' + (f' - {account}' if account else ''))
        ax1.set_ylabel('Trading Fees (USDT)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Funding Fees (USDT)')
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.xticks(rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        return fig
    
    def monthly_performance_heatmap(self, account=None, figsize=(12, 8)):
        """
        Create a heatmap showing monthly performance
        
        Args:
            account (str, optional): Specific account to analyze
            figsize (tuple, optional): Figure size
        """
        if self.data is None:
            self.load_data(account=account)
            
        # Calculate returns
        returns_df = self.calculate_returns(account=account)
        
        if account:
            returns_df = returns_df[returns_df['account_name'] == account]
            
        # Extract month and year from date
        returns_df['year'] = returns_df['date'].dt.year
        returns_df['month'] = returns_df['date'].dt.month
        
        # Create a pivot table of monthly returns for each account
        account_heatmaps = {}
        
        for account_name in returns_df['account_name'].unique():
            account_returns = returns_df[returns_df['account_name'] == account_name]
            
            # Calculate monthly returns
            monthly_returns = account_returns.groupby(['year', 'month'])['daily_return'].apply(
                lambda x: (1 + x).prod() - 1
            ).reset_index()
            
            # Create pivot table
            pivot_table = monthly_returns.pivot(index='year', columns='month', values='daily_return')
            
            # Store for this account
            account_heatmaps[account_name] = pivot_table
        
        # Create plots
        figs = []
        
        for account_name, pivot_table in account_heatmaps.items():
            fig, ax = plt.subplots(figsize=figsize)
            
            # Use seaborn heatmap
            hm = sns.heatmap(
                pivot_table * 100,  # Convert to percentage
                ax=ax,
                annot=True,
                fmt=".2f",
                cmap="RdYlGn",
                center=0,
                linewidths=1,
                cbar_kws={'label': 'Monthly Return (%)'}
            )
            
            # Format the plot
            ax.set_title(f'Monthly Returns Heatmap - {account_name}')
            ax.set_ylabel('Year')
            ax.set_xlabel('Month')
            
            # Get the current x-tick locations and labels
            current_ticks = ax.get_xticks()
            
            # Replace month numbers with names only if there are exactly 12 columns
            if len(current_ticks) == 12:
                month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                            'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                ax.set_xticklabels(month_names, rotation=0)
            else:
                # For less than 12 months, use the actual month numbers from the pivot table
                actual_months = pivot_table.columns.tolist()
                month_abbr = {
                    1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                    7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
                }
                labels = [month_abbr.get(m, str(m)) for m in actual_months]
                ax.set_xticklabels(labels, rotation=0)
            
            plt.tight_layout()
            figs.append(fig)
            
        return figs
    
    def performance_distribution(self, account=None, days=None, figsize=(12, 6)):
        """
        Create a histogram showing the distribution of daily returns
        
        Args:
            account (str, optional): Specific account to analyze
            days (int, optional): Number of days to look back
            figsize (tuple, optional): Figure size
        """
        # Calculate returns
        returns_df = self.calculate_returns(account=account)
        
        if days:
            cutoff_date = (datetime.now() - timedelta(days=days)).date()
            returns_df = returns_df[returns_df['date'].dt.date >= cutoff_date]
            
        if account:
            returns_df = returns_df[returns_df['account_name'] == account]
            
        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)
        
        for account_name in returns_df['account_name'].unique():
            account_returns = returns_df[returns_df['account_name'] == account_name]
            daily_returns = account_returns['daily_return'].dropna() * 100  # Convert to percentage
            
            # Skip if not enough data
            if len(daily_returns) < 5:
                continue
                
            # Plot histogram with kernel density estimate
            sns.histplot(daily_returns, kde=True, label=account_name, alpha=0.6, ax=ax)
            
            # Add mean line
            ax.axvline(daily_returns.mean(), color='r', linestyle='--', 
                      label=f'{account_name} Mean: {daily_returns.mean():.2f}%')
                      
        # Format the plot
        ax.set_title('Daily Returns Distribution' + (f' - {account}' if account else ''))
        ax.set_xlabel('Daily Return (%)')
        ax.set_ylabel('Frequency')
        
        # Add zero line
        ax.axvline(0, color='k', linestyle='-', alpha=0.3)
        
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return fig
    
    def generate_report(self, account=None, days=None, output_dir=REPORTS_DIR):
        """
        Generate a comprehensive analysis report
        
        Args:
            account (str, optional): Specific account to analyze
            days (int, optional): Number of days to look back
            output_dir (str, optional): Directory to save report
        """
        # Load data if not already loaded
        if self.data is None:
            self.load_data(days=days, account=account)
            
        # Create output directory for this report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_name = f"analysis_{account or 'all'}"
        if days:
            report_name += f"_{days}days"
        report_name += f"_{timestamp}"
        
        report_dir = os.path.join(output_dir, report_name)
        os.makedirs(report_dir, exist_ok=True)
        
        # Generate plots with error handling
        plots = {}
        try:
            plots['equity_curve'] = self.create_equity_curve(account=account, days=days)
        except Exception as e:
            print(f"Error generating equity curve: {e}")
        
        try:
            plots['drawdown'] = self.create_drawdown_chart(account=account, days=days)
        except Exception as e:
            print(f"Error generating drawdown chart: {e}")
            
        try:
            plots['exposure'] = self.analyze_exposure(account=account, days=days)
        except Exception as e:
            print(f"Error generating exposure analysis: {e}")
            
        try:
            plots['trading_activity'] = self.analyze_trading_activity(account=account, days=days)
        except Exception as e:
            print(f"Error generating trading activity: {e}")
            
        try:
            plots['fees'] = self.analyze_fees(account=account, days=days)
        except Exception as e:
            print(f"Error generating fee analysis: {e}")
            
        try:
            plots['performance_dist'] = self.performance_distribution(account=account, days=days)
        except Exception as e:
            print(f"Error generating performance distribution: {e}")
        
        # Save plots
        for name, fig in plots.items():
            try:
                fig.savefig(os.path.join(report_dir, f"{name}.png"), dpi=300, bbox_inches='tight')
                plt.close(fig)
            except Exception as e:
                print(f"Error saving {name} plot: {e}")
            
        # Generate monthly heatmaps with error handling
        try:
            heatmaps = self.monthly_performance_heatmap(account=account)
            for i, fig in enumerate(heatmaps):
                fig.savefig(os.path.join(report_dir, f"monthly_heatmap_{i}.png"), dpi=300, bbox_inches='tight')
                plt.close(fig)
        except Exception as e:
            print(f"Error generating monthly heatmaps: {e}")
            
        # Calculate metrics with error handling
        try:
            returns_df = self.calculate_returns(account=account)
            if days:
                cutoff_date = (datetime.now() - timedelta(days=days)).date()
                returns_df = returns_df[returns_df['date'].dt.date >= cutoff_date]
                
            metrics = self.calculate_metrics(returns_df=returns_df, account=account)
            
            # Create metrics summary
            metrics_df = pd.DataFrame.from_dict(metrics, orient='index')
            
            # Format metrics for readability
            formatted_metrics = metrics_df.copy()
            for col in formatted_metrics.columns:
                if col in ['total_return', 'annualized_return', 'volatility', 'sharpe_ratio', 
                          'max_drawdown', 'win_rate', 'best_day', 'worst_day', 'avg_gain', 'avg_loss']:
                    formatted_metrics[col] = formatted_metrics[col].apply(lambda x: f"{x*100:.2f}%")
                elif col in ['profit_factor']:
                    formatted_metrics[col] = formatted_metrics[col].apply(lambda x: f"{x:.2f}")
                elif col in ['last_equity']:
                    formatted_metrics[col] = formatted_metrics[col].apply(lambda x: f"{x:,.2f}")
                    
            # Save metrics to CSV
            metrics_df.to_csv(os.path.join(report_dir, "metrics.csv"))
            
            # Generate HTML report
            self._generate_html_report(report_dir, formatted_metrics, account, days)
        except Exception as e:
            print(f"Error calculating metrics: {e}")
            # Generate HTML report even if metrics fail
            self._generate_html_report(report_dir, pd.DataFrame(), account, days)
        
        print(f"Report generated: {report_dir}")
        return report_dir
    
    def _generate_html_report(self, report_dir, metrics_df, account=None, days=None):
        """
        Generate HTML report from analysis results
        
        Args:
            report_dir (str): Directory containing report files
            metrics_df (DataFrame): Metrics dataframe
            account (str, optional): Specific account analyzed
            days (int, optional): Number of days analyzed
        """
        # Check if metrics_df is empty
        if metrics_df.empty:
            metrics_html = "<p>No metrics available - not enough data to calculate.</p>"
        else:
            # Create table header
            metrics_html = """
            <table>
                <tr>
                    <th>Account</th>
                    <th>Total Days</th>
                    <th>Total Return</th>
                    <th>Annualized Return</th>
                    <th>Volatility</th>
                    <th>Sharpe Ratio</th>
                    <th>Max Drawdown</th>
                    <th>Win Rate</th>
                """
            
            # Add profit factor column if available
            if 'profit_factor' in metrics_df.columns:
                metrics_html += "<th>Profit Factor</th>\n"
                
            # Add last equity column if available
            if 'last_equity' in metrics_df.columns:
                metrics_html += "<th>Last Equity</th>\n"
                
            metrics_html += "</tr>\n"
            
            # Add metrics rows
            for index, row in metrics_df.iterrows():
                metrics_html += f"""
                <tr>
                    <td>{index}</td>
                    <td>{row.get('total_days', 'N/A')}</td>
                    <td>{row.get('total_return', 'N/A')}</td>
                    <td>{row.get('annualized_return', 'N/A')}</td>
                    <td>{row.get('volatility', 'N/A')}</td>
                    <td>{row.get('sharpe_ratio', 'N/A')}</td>
                    <td>{row.get('max_drawdown', 'N/A')}</td>
                    <td>{row.get('win_rate', 'N/A')}</td>
                """
                
                # Add profit factor if available
                if 'profit_factor' in row:
                    metrics_html += f"<td>{row['profit_factor']}</td>\n"
                    
                # Add last equity if available
                if 'last_equity' in row:
                    metrics_html += f"<td>{row['last_equity']}</td>\n"
                    
                metrics_html += "</tr>\n"
                
            metrics_html += "</table>\n"
        
        # Create HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Trading Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2 {{ color: #2a5e35; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ text-align: left; padding: 8px; }}
                th {{ background-color: #2a5e35; color: white; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
                .plot-container {{ margin-bottom: 30px; }}
                .plot-container img {{ max-width: 100%; }}
            </style>
        </head>
        <body>
            <h1>Trading Analysis Report</h1>
            <p><strong>Generated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            <p><strong>Account:</strong> {account or "All Accounts"}</p>
            <p><strong>Period:</strong> {f"Last {days} days" if days else "All available data"}</p>
            
            <h2>Performance Metrics</h2>
            {metrics_html}
        """
        
        # Check if each plot file exists before adding it to the HTML
        plot_sections = [
            ("Equity Curve", "equity_curve.png"),
            ("Drawdown Analysis", "drawdown.png"),
            ("Net Exposure", "exposure.png"),
            ("Trading Activity", "trading_activity.png"),
            ("Fee Analysis", "fees.png"),
            ("Daily Returns Distribution", "performance_dist.png")
        ]
        
        for title, filename in plot_sections:
            if os.path.exists(os.path.join(report_dir, filename)):
                html_content += f"""
                <h2>{title}</h2>
                <div class="plot-container">
                    <img src="{filename}" alt="{title}">
                </div>
                """
        
        # Add monthly heatmaps if they exist
        heatmap_files = [f for f in os.listdir(report_dir) if f.startswith("monthly_heatmap_")]
        if heatmap_files:
            html_content += """
            <h2>Monthly Performance Heatmaps</h2>
            """
            
            for heatmap_file in sorted(heatmap_files):
                html_content += f"""
                <div class="plot-container">
                    <img src="{heatmap_file}" alt="Monthly Performance Heatmap">
                </div>
                """
                
        html_content += """
        </body>
        </html>
        """
        
        # Write HTML to file
        with open(os.path.join(report_dir, "report.html"), "w") as f:
            f.write(html_content)
            
def main():
    """Main function to run the analysis"""
    parser = argparse.ArgumentParser(description='Trading Data Analysis')
    parser.add_argument('--account', type=str, help='Specific account to analyze')
    parser.add_argument('--days', type=int, help='Number of days to look back')
    parser.add_argument('--db', type=str, default=DB_PATH, help='Path to database file')
    parser.add_argument('--output', type=str, default=REPORTS_DIR, help='Output directory for reports')
    parser.add_argument('--metrics-only', action='store_true', help='Only calculate metrics, no plots')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = TradingAnalyzer(db_path=args.db)
    
    if args.metrics_only:
        # Just print metrics
        returns_df = analyzer.calculate_returns(account=args.account)
        if args.days:
            cutoff_date = (datetime.now() - timedelta(days=args.days)).date()
            returns_df = returns_df[returns_df['date'].dt.date >= cutoff_date]
        
        metrics = analyzer.calculate_metrics(returns_df=returns_df, account=args.account)
        metrics_df = pd.DataFrame.from_dict(metrics, orient='index')
        print(metrics_df)
    else:
        # Generate full report
        analyzer.generate_report(account=args.account, days=args.days, output_dir=args.output)

def analyze_trading_pairs(analyzer, account=None, days=None, figsize=(12, 8)):
    """
    Analyze which trading pairs are most profitable
    
    Args:
        analyzer: TradingAnalyzer instance
        account (str, optional): Specific account to analyze
        days (int, optional): Number of days to look back
        figsize (tuple, optional): Figure size
    """
    # This would require additional data not in the current database schema
    # You could store this data in another table and process it here
    pass

def analyze_long_short_performance(analyzer, account=None, days=None, figsize=(12, 8)):
    """
    Compare performance of long vs short positions
    
    Args:
        analyzer: TradingAnalyzer instance
        account (str, optional): Specific account to analyze
        days (int, optional): Number of days to look back
        figsize (tuple, optional): Figure size
    """
    # This would require additional data not in the current database schema
    # You could store this data in another table and process it here
    pass

def analyze_market_correlation(analyzer, account=None, days=None, figsize=(12, 8)):
    """
    Analyze correlation between account performance and market benchmarks
    
    Args:
        analyzer: TradingAnalyzer instance
        account (str, optional): Specific account to analyze
        days (int, optional): Number of days to look back
        figsize (tuple, optional): Figure size
    """
    # This would require additional data on market benchmarks
    # You could download this data and process it here
    pass

def backtest_strategy(analyzer, strategy_params, account=None, days=None):
    """
    Backtest a trading strategy against historical data
    
    Args:
        analyzer: TradingAnalyzer instance
        strategy_params: Parameters for the strategy
        account (str, optional): Specific account to analyze
        days (int, optional): Number of days to look back
    """
    # This would require more detailed data about individual trades
    # You could implement a backtesting framework here
    pass

def analyze_volatility_regime(analyzer, account=None, days=None, window=20, figsize=(12, 8)):
    """
    Analyze performance during different volatility regimes
    
    Args:
        analyzer: TradingAnalyzer instance
        account (str, optional): Specific account to analyze
        days (int, optional): Number of days to look back
        window (int): Rolling window for volatility calculation
        figsize (tuple, optional): Figure size
    """
    if analyzer.data is None:
        analyzer.load_data(days=days, account=account)
    
    returns_df = analyzer.calculate_returns(account=account)
    
    if days:
        cutoff_date = (datetime.now() - timedelta(days=days)).date()
        returns_df = returns_df[returns_df['date'].dt.date >= cutoff_date]
    
    if account:
        returns_df = returns_df[returns_df['account_name'] == account]
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
    
    for account_name in returns_df['account_name'].unique():
        account_returns = returns_df[returns_df['account_name'] == account_name]
        
        # Skip if not enough data
        if len(account_returns) <= window:
            continue
        
        # Calculate rolling volatility
        account_returns['rolling_vol'] = account_returns['daily_return'].rolling(window=window).std() * np.sqrt(252) * 100
        
        # Plot rolling volatility
        ax1.plot(account_returns['date'], account_returns['rolling_vol'], label=f"{account_name} - {window}-day Vol")
        
        # Calculate cumulative return
        account_returns['cum_return'] = (1 + account_returns['daily_return']).cumprod() - 1
        
        # Plot cumulative return
        ax2.plot(account_returns['date'], account_returns['cum_return'] * 100, label=f"{account_name} - Return")
    
    # Format the plots
    ax1.set_title('Volatility Regime Analysis' + (f' - {account}' if account else ''))
    ax1.set_ylabel('Annualized Volatility (%)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Cumulative Return (%)')
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.xticks(rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return fig

def analyze_consecutive_wins_losses(analyzer, account=None, days=None, figsize=(12, 6)):
    """
    Analyze consecutive wins and losses
    
    Args:
        analyzer: TradingAnalyzer instance
        account (str, optional): Specific account to analyze
        days (int, optional): Number of days to look back
        figsize (tuple, optional): Figure size
    """
    returns_df = analyzer.calculate_returns(account=account)
    
    if days:
        cutoff_date = (datetime.now() - timedelta(days=days)).date()
        returns_df = returns_df[returns_df['date'].dt.date >= cutoff_date]
    
    if account:
        returns_df = returns_df[returns_df['account_name'] == account]
    
    results = {}
    
    for account_name in returns_df['account_name'].unique():
        account_returns = returns_df[returns_df['account_name'] == account_name]
        daily_returns = account_returns['daily_return'].dropna()
        
        # Skip if not enough data
        if len(daily_returns) < 5:
            continue
        
        # Categorize returns as wins, losses, or flat
        wins = (daily_returns > 0).astype(int)
        losses = (daily_returns < 0).astype(int)
        
        # Count consecutive wins and losses
        win_streaks = []
        loss_streaks = []
        
        current_win_streak = 0
        current_loss_streak = 0
        
        for win, loss in zip(wins, losses):
            if win:
                # Reset loss streak
                if current_loss_streak > 0:
                    loss_streaks.append(current_loss_streak)
                    current_loss_streak = 0
                
                # Increment win streak
                current_win_streak += 1
            elif loss:
                # Reset win streak
                if current_win_streak > 0:
                    win_streaks.append(current_win_streak)
                    current_win_streak = 0
                
                # Increment loss streak
                current_loss_streak += 1
            else:
                # Flat day, reset both
                if current_win_streak > 0:
                    win_streaks.append(current_win_streak)
                    current_win_streak = 0
                if current_loss_streak > 0:
                    loss_streaks.append(current_loss_streak)
                    current_loss_streak = 0
        
        # Add final streaks if they exist
        if current_win_streak > 0:
            win_streaks.append(current_win_streak)
        if current_loss_streak > 0:
            loss_streaks.append(current_loss_streak)
        
        # Calculate statistics
        results[account_name] = {
            'max_consecutive_wins': max(win_streaks) if win_streaks else 0,
            'max_consecutive_losses': max(loss_streaks) if loss_streaks else 0,
            'avg_win_streak': np.mean(win_streaks) if win_streaks else 0,
            'avg_loss_streak': np.mean(loss_streaks) if loss_streaks else 0,
            'win_streaks': win_streaks,
            'loss_streaks': loss_streaks
        }
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot histograms of streaks
    for account_name, account_results in results.items():
        win_streaks = account_results['win_streaks']
        loss_streaks = account_results['loss_streaks']
        
        if win_streaks:
            sns.histplot(win_streaks, label=f"{account_name} - Win Streaks", 
                        alpha=0.6, color='green', kde=False, discrete=True, ax=ax)
        if loss_streaks:
            sns.histplot(loss_streaks, label=f"{account_name} - Loss Streaks", 
                        alpha=0.6, color='red', kde=False, discrete=True, ax=ax)
    
    # Format the plot
    ax.set_title('Analysis of Consecutive Wins and Losses' + (f' - {account}' if account else ''))
    ax.set_xlabel('Streak Length')
    ax.set_ylabel('Frequency')
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return fig, results

class AdvancedTradingAnalyzer(TradingAnalyzer):
    """Extended class with advanced analysis capabilities"""
    
    def __init__(self, db_path=DB_PATH):
        super().__init__(db_path)
    
    def calculate_advanced_metrics(self, returns_df=None, account=None):
        """
        Calculate advanced trading metrics
        
        Args:
            returns_df (DataFrame, optional): DataFrame with daily returns
            account (str, optional): Specific account to analyze
        """
        if returns_df is None:
            returns_df = self.calculate_returns(account=account)
            
        if account:
            returns_df = returns_df[returns_df['account_name'] == account]
            
        metrics = {}
        
        for account_name in returns_df['account_name'].unique():
            account_returns = returns_df[returns_df['account_name'] == account_name]
            
            # Daily metrics
            daily_returns = account_returns['daily_return'].dropna()
            
            # Skip if not enough data
            if len(daily_returns) < 10:
                continue
                
            # Calculate basic metrics first
            basic_metrics = {
                'total_days': len(daily_returns),
                'total_return': (1 + daily_returns).prod() - 1,
                'annualized_return': (1 + daily_returns).prod() ** (252 / len(daily_returns)) - 1,
                'volatility': daily_returns.std() * np.sqrt(252),
                'win_rate': (daily_returns > 0).mean(),
            }
            
            # Now calculate advanced metrics
            
            # Sortino ratio (using downside deviation instead of total volatility)
            downside_returns = daily_returns[daily_returns < 0]
            downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
            sortino_ratio = (daily_returns.mean() * 252) / downside_deviation if downside_deviation > 0 else float('inf')
            
            # Calmar ratio (annualized return / max drawdown)
            # Calculate drawdowns
            cum_returns = (1 + daily_returns).cumprod()
            running_max = cum_returns.cummax()
            drawdowns = (cum_returns / running_max - 1)
            max_drawdown = drawdowns.min()
            calmar_ratio = basic_metrics['annualized_return'] / abs(max_drawdown) if max_drawdown < 0 else float('inf')
            
            # Omega ratio (probability weighted ratio of gains versus losses)
            threshold = 0  # Can be set to risk-free rate or other threshold
            gains = daily_returns[daily_returns > threshold] - threshold
            losses = threshold - daily_returns[daily_returns < threshold]
            omega_ratio = gains.sum() / losses.sum() if losses.sum() > 0 else float('inf')
            
            # Skewness and kurtosis
            skewness = stats.skew(daily_returns)
            kurtosis = stats.kurtosis(daily_returns)
            
            # Advanced metrics dictionary
            advanced_metrics = {
                'sortino_ratio': sortino_ratio,
                'calmar_ratio': calmar_ratio,
                'omega_ratio': omega_ratio,
                'skewness': skewness,
                'kurtosis': kurtosis,
                'average_gain': daily_returns[daily_returns > 0].mean() if len(daily_returns[daily_returns > 0]) > 0 else 0,
                'average_loss': daily_returns[daily_returns < 0].mean() if len(daily_returns[daily_returns < 0]) > 0 else 0,
                'gain_to_loss_ratio': abs(daily_returns[daily_returns > 0].mean() / daily_returns[daily_returns < 0].mean()) if len(daily_returns[daily_returns < 0]) > 0 and daily_returns[daily_returns < 0].mean() != 0 else float('inf'),
                'max_consecutive_days_up': self._max_consecutive_days(daily_returns, condition=lambda x: x > 0),
                'max_consecutive_days_down': self._max_consecutive_days(daily_returns, condition=lambda x: x < 0),
            }
            
            # Combine metrics
            metrics[account_name] = {**basic_metrics, **advanced_metrics}
            
        return metrics
    
    def _max_consecutive_days(self, returns, condition):
        """Helper method to calculate max consecutive days meeting a condition"""
        # Convert the returns to a binary array based on the condition
        binary = returns.apply(condition).astype(int)
        
        # If no days meet the condition, return 0
        if binary.sum() == 0:
            return 0
            
        # Calculate consecutive days
        consecutive = binary.groupby(binary.ne(binary.shift()).cumsum()).sum()
        
        # Return the maximum
        return consecutive.max()
    
    def monte_carlo_simulation(self, account=None, days=None, n_simulations=1000, projection_days=252, figsize=(12, 8)):
        """
        Perform Monte Carlo simulation to project future performance
        
        Args:
            account (str): Account to analyze (required)
            days (int, optional): Number of days to use for historical data
            n_simulations (int): Number of simulations to run
            projection_days (int): Number of days to project forward
            figsize (tuple): Figure size
        """
        if account is None:
            raise ValueError("Account must be specified for Monte Carlo simulation")
            
        returns_df = self.calculate_returns(account=account)
        
        if days:
            cutoff_date = (datetime.now() - timedelta(days=days)).date()
            returns_df = returns_df[returns_df['date'].dt.date >= cutoff_date]
            
        account_returns = returns_df[returns_df['account_name'] == account]
        daily_returns = account_returns['daily_return'].dropna()
        
        # Skip if not enough data
        if len(daily_returns) < 30:
            raise ValueError(f"Not enough data for account {account}. Need at least 30 data points.")
            
        # Get current equity
        current_equity = account_returns['equity'].iloc[-1]
        
        # Perform Monte Carlo simulation
        np.random.seed(42)  # For reproducibility
        
        # Set up empty array for simulations
        simulations = np.zeros((projection_days, n_simulations))
        
        # Initialize starting point
        simulations[0, :] = current_equity
        
        # Loop through projection days
        for i in range(1, projection_days):
            # Random sample from historical returns
            random_returns = np.random.choice(daily_returns, size=n_simulations, replace=True)
            simulations[i, :] = simulations[i-1, :] * (1 + random_returns)
            
        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot simulations
        for i in range(min(100, n_simulations)):  # Plot max 100 lines for clarity
            ax.plot(simulations[:, i], color='blue', alpha=0.1)
            
        # Plot mean
        ax.plot(simulations.mean(axis=1), color='red', linewidth=2, label='Mean')
        
        # Plot quantiles
        quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
        quantile_data = np.quantile(simulations, q=quantiles, axis=1)
        
        for i, q in enumerate(quantiles):
            ax.plot(quantile_data[i, :], color=f'C{i+1}', linewidth=1.5, linestyle='--', 
                   label=f'{q*100}% Quantile')
            
        # Format the plot
        ax.set_title(f'Monte Carlo Simulation - {account} ({n_simulations} Simulations, {projection_days} Days)')
        ax.set_xlabel('Days')
        ax.set_ylabel('Equity (USDT)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Format y-axis to show thousands with K
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x/1000:.1f}K'))
        
        plt.tight_layout()
        
        # Calculate projected metrics
        final_equity = simulations[-1, :]
        projected_return = (final_equity / current_equity - 1) * 100
        
        projected_metrics = {
            'mean_projected_return': projected_return.mean(),
            'median_projected_return': np.median(projected_return),
            'min_projected_return': projected_return.min(),
            'max_projected_return': projected_return.max(),
            '10th_percentile': np.percentile(projected_return, 10),
            '25th_percentile': np.percentile(projected_return, 25),
            '75th_percentile': np.percentile(projected_return, 75),
            '90th_percentile': np.percentile(projected_return, 90),
            'probability_of_profit': (projected_return > 0).mean()
        }
        
        return fig, projected_metrics, simulations

if __name__ == "__main__":
    main()