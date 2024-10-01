import csv
from datetime import datetime
from pybit.unified_trading import HTTP
import os
from dotenv import dotenv_values
import ccxt
from main import get_accounts_from_env

# File to store deposit/withdrawal data
TRANSACTIONS_FILE = "transactions.csv"

def choose_account(accounts):
    print("Available accounts:")
    for account in accounts:
        print(f"{account['id']}: {account['name']}")
    while True:
        choice = input("Enter the account ID you want to use: ")
        selected_account = next((acc for acc in accounts if acc['id'] == choice), None)
        if selected_account:
            return selected_account
        else:
            print("Invalid account ID. Please try again.")

def log_transaction(account_id, account_name, amount, transaction_type):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(TRANSACTIONS_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([timestamp, account_id, account_name, amount, transaction_type])

def flows_manager(accounts):
    while True:
        selected_account = choose_account(accounts)
        if not selected_account:
            print("Exiting flow manager.")
            break

        print(f"\nSelected account: {selected_account['name']} (ID: {selected_account['id']})")
        
        while True:
            print("\nWhat would you like to do?")
            print("1. Log Deposit")
            print("2. Log Withdrawal")
            print("3. Select Different Account")
            print("4. Exit")
            choice = input("Enter your choice: ")
            
            if choice == '1':
                amount = float(input("Enter deposit amount (USDT): "))
                log_transaction(selected_account['id'], selected_account['name'], amount, 'deposit')
            elif choice == '2':
                amount = float(input("Enter withdrawal amount (USDT): "))
                log_transaction(selected_account['id'], selected_account['name'], amount, 'withdrawal')
            elif choice == '3':
                break  # This will go back to account selection
            elif choice == '4':
                return  # This will exit the entire flows_manager function
            else:
                print("Invalid choice. Please try again.")

def main():
    if not os.path.exists(TRANSACTIONS_FILE):
        with open(TRANSACTIONS_FILE, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Timestamp', 'Account ID', 'Account Name', 'Amount', 'Type'])
    accounts = get_accounts_from_env()
    if not accounts:
        print("No accounts found in .env file.")
    else:
        flows_manager(accounts)

if __name__ == "__main__":
    main()