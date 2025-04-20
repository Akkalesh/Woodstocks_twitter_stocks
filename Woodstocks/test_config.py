from config import TWITTER_CREDS, STOCK_SYMBOL

print("\n🔧 Configuration Test Results:")
print("-" * 30)
print(f"Stock Symbol: {STOCK_SYMBOL}")
print(f"Twitter Key (first 4 chars): {TWITTER_CREDS['api_key'][:4]}...")
print("-" * 30)