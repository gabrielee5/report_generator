# REPORT GENERATOR
Generates a trading report periodically.

For now the backup files work and there is a small description of what they do at the beginning of every of them.

It should run periodically in automatic when correctly set up using a cron job.

## Instructions
The file test.env is a template for the structure of the .env file.

The main.py file works fine and stores the data in a database. Currently working on a new and more efficient version.

## File .env
The structure of the .env file should be this:

    001_api_key = "abc"
    001_api_secret = "xyz"
    001_name = "account1"


## TO DO
The positions_analysis file needs some adjustments as it doesnt do what it is supposted to.