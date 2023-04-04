import requests
import pandas as pd
import json

# Define the API endpoint URL
url = "https://api.alternative.me/fng/?limit=0&format=json"

# Send a GET request to the API endpoint URL
response = requests.get(url)

data = json.loads(response.text)
df = pd.DataFrame(data["data"])
df = df[["timestamp", "value"]]
df["date"] = pd.to_datetime(df["timestamp"], unit="s").dt.date
df.drop(columns=["timestamp"], inplace=True)

df = df.loc[df["date"] >= pd.to_datetime("2023-02-22")]
df = df.loc[df["date"] <= pd.to_datetime("2023-03-13")]

df.to_csv("./test/fear_greed_test.csv", index=False)

