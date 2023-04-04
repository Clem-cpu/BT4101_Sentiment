import requests
import pandas as pd
import json
import time
import datetime
from datetime import timezone

def get_btc_data():
    # Define the endpoint, interval, and symbol
    endpoint = "https://api.binance.com/api/v3/klines"
    interval = "1d"
    symbol = "BTCUSDT"


    start_date = "2022-01-01 00:00:00"
    start_timestamp = (
        int(time.mktime(time.strptime(start_date, "%Y-%m-%d %H:%M:%S"))) * 1000
    )

    # Getting the current date
    # and time
    dt = datetime.datetime.now(timezone.utc)
    utc_time = dt.replace(tzinfo=timezone.utc)
    utc_timestamp = utc_time.timestamp()
    end_timestamp = int(utc_timestamp * 1000)


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
    df.to_csv("./test/application_files/bitcoin_prices.csv", index=False)




def market_cron():
    # Define the API endpoint URL
    url = "https://api.alternative.me/fng/?limit=0&format=json"

    # Send a GET request to the API endpoint URL
    response = requests.get(url)

    data = json.loads(response.text)
    df = pd.DataFrame(data["data"])
    df = df[["timestamp", "value"]]
    df["date"] = pd.to_datetime(df["timestamp"], unit="s").dt.date
    df = df.loc[df["date"] >= pd.to_datetime("2022-01-01")]
    df.drop(columns=["timestamp"], inplace=True)
    df.sort_values(by=["date"], inplace=True)
    df.to_csv("./test/application_files/fear_greed.csv", index=False)
    get_btc_data()
