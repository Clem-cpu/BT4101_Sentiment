import pandas as pd
import numpy as np
from datetime import datetime
from tensorflow import keras
import tensorflow as tf
from sklearn.metrics import accuracy_score
from numpy.random import seed
import joblib
from keras.models import load_model


df = pd.read_csv("./test/bitcoin_prices_2023.csv")
df_fear_greed = pd.read_csv("./test/fear_greed_test.csv")
df_reddit = pd.read_csv("./test/reddit_scraper/reddit_sentiment.csv")
df_news = pd.read_csv("./test/news_articles/news_sentiment_test.csv")


### Data cleaning for bitcoin prices
df.rename(columns={"Open time": "date"}, inplace=True)
df["date"] = df["date"].apply(
    lambda x: datetime.fromtimestamp(x / 1000).strftime("%Y-%m-%d %H:%M:%S")
)
df["date"] = pd.to_datetime(df["date"]).dt.date
df["date"] = df["date"].astype("datetime64[ns]")
df.drop(columns=["Ignore"], inplace=True)


### Data cleaning for fear greed index
df_fear_greed.rename(columns={"value": "fear_greed"}, inplace=True)
df_fear_greed["date"] = df_fear_greed["date"].astype("datetime64[ns]")


### Data cleaning for reddit data
df_reddit["date"] = df_reddit["date"].astype("datetime64[ns]")


### Data cleaning for news data
df_news["date"] = df_news["date"].astype("datetime64[ns]")

### Merging datasets
df = pd.merge(df, df_reddit, how="left", on="date")
df = pd.merge(df, df_fear_greed, how="left", on="date")
df = pd.merge(df, df_news, how="left", on="date")
df = df.drop(df.select_dtypes(include=["datetime"]).columns, axis=1)

df["Change"] = (df["Close"] > df["Close"].shift(1)).astype(int)
df.dropna(inplace=True)
df["text_positive"] = 0


# Normalize the data using MinMaxScaler
scaler = joblib.load("./test/save_files/scaler.gz")
split = int(round(len(df) / 100 * 70, 0))
test = df
test_set = scaler.transform(test)

# Convert the data into a 3D format suitable for LSTM input
def create_dataset(dataset, look_back, column_index):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i : (i + look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back, column_index])

    return np.array(dataX), np.array(dataY)


def train_model_lookback(look_back: int):
    column_index = df.columns.get_loc("Change")
    X_test, y_test = create_dataset(test_set, look_back, column_index)

    X_test = np.reshape(X_test, (X_test.shape[0], look_back, X_test.shape[2]))

    model = load_model(f"./test/save_files/{look_back}_cnn_model")

    # Make predictions on the test data
    predictions = model.predict(X_test)
    testPredict_proba = model.predict(X_test)

    # Round the predictions to 0 or 1 based on their direction of price movement
    testPredict = np.round(testPredict_proba)

    # Calculate the accuracy of the model
    testAccuracy = accuracy_score(y_test, testPredict.flatten())

    print(f"Lag is {look_back}")
    print(f"test accuracy: {testAccuracy}")


train_model_lookback(1)
train_model_lookback(3)
train_model_lookback(5)
