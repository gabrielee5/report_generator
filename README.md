# REPORT GENERATOR
Generates a trading report periodically.

For now the backup files work and there is a small description of what they do at the beginning of every of them.

The main file needs to be run in aws but I haven't tested it yet in the aws environment. It should run periodically in automatic when correctly set up.

## Instructions
The file test.env is a template for the structure of the .env file.

The file backup_4.py run perfectly and is meant to generate a daily report when activated; also it stores daily data in a database and it creates a log file.
The other backup files are previous versions, so they are less sophisticated.

## File .env
The structure of the .env file should be this:

    001_api_key = "abc"
    001_api_secret = "xyz"
    001_name = "account1"