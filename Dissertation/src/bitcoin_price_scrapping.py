import requests
import pandas as pd
import json
import time

# Define the endpoint, interval, and symbol
endpoint = "https://api.binance.com/api/v3/klines"
interval = "1d"
symbol = "BTCUSDT"


start_date = "2023-02-22 00:00:00"
start_timestamp = (
    int(time.mktime(time.strptime(start_date, "%Y-%m-%d %H:%M:%S"))) * 1000
)

end_date = "2023-03-14 00:00:00"
end_timestamp = int(time.mktime(time.strptime(end_date, "%Y-%m-%d %H:%M:%S"))) * 1000

# Send the request to the endpoint
response = requests.get(
    endpoint,
    params={
        "interval": interval,
        "symbol": symbol,
        "startTime": start_timestamp,
        "endTime": end_timestamp,
    },
)

# Parse the JSON response
data = json.loads(response.text)

# Create a DataFrame from the data
df = pd.DataFrame(
    data,
    columns=[
        "Open time",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "Close time",
        "Quote asset volume",
        "Number of trades",
        "Taker buy base asset volume",
        "Taker buy quote asset volume",
        "Ignore",
    ],
)

# Write the DataFrame to a CSV file
print(df.tail(5))
print(df.info())
df.to_csv("./test/bitcoin_prices_2023.csv", index=False)
print("Data written to bitcoin_prices.csv")
