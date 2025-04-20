

import os
from dotenv import load_dotenv

load_dotenv()

# Twitter API
TWITTER_CREDS = {
    'api_key': os.getenv('TWITTER_API_KEY'),
    'api_secret': os.getenv('TWITTER_API_SECRET'),
    'access_token': os.getenv('TWITTER_ACCESS_TOKEN'),
    'access_secret': os.getenv('TWITTER_ACCESS_SECRET'),
    'bearer_token': os.getenv('BEARER_TOKEN')
}

# Project Settings
STOCK_SYMBOL = "HDFCBANK.NS"  # Change to ICICIBANK.NS/SBIN.NS etc.
LOOKBACK_DAYS = 60