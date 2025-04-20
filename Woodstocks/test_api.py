import requests
import json

# 1. First test the root endpoint
response = requests.get("http://localhost:5000/")
print("Root endpoint:", response.text)

# 2. Prepare test data - MODIFY THIS TO MATCH YOUR MODEL'S REQUIREMENTS
test_data = {
    "High": 255.50,
    "Low": 248.20,
    "Open": 252.30,
    "Close": 250.00,
    "Volume": 5000000,
    "daily_return": 0.015,
    "avg_sentiment": 0.15,
    "tweet_count": 85,
    "5_day_ma": 250.50,
    "10_day_volatility": 0.02,
    "sentiment_lag1": 0.12,
    "sentiment_ma3": 0.14
}

# 3. Make the prediction request
try:
    print("\nMaking prediction request...")
    response = requests.post(
        "http://localhost:5000/predict",
        headers={"Content-Type": "application/json"},
        data=json.dumps(test_data)
    )
    
    print("Status code:", response.status_code)
    print("Response:")
    print(json.dumps(response.json(), indent=2))
    
except Exception as e:
    print("Error:", e)