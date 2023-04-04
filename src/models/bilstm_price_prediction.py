import pandas as pd
import numpy as np
from datetime import datetime
from keras.layers import Dropout, LSTM, Dense, Bidirectional
from keras.models import Sequential
from keras import layers
from tensorflow import keras
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


df = pd.read_csv("./test/bitcoin_prices_2022.csv")
df_fear_greed = pd.read_csv("./test/fear_and_greed_index.csv")
df_reddit = pd.read_csv("./test/reddit/reddit_sentiment.csv")
df_news = pd.read_csv("./test/news_articles/news_sentiment.csv")


### Data cleaning for bitcoin prices

df.rename(columns={"Open time": "date"}, inplace=True)
df["date"] = df["date"].apply(
    lambda x: datetime.fromtimestamp(x / 1000).strftime("%Y-%m-%d %H:%M:%S")
)
df["date"] = pd.to_datetime(df["date"]).dt.date
df["date"] = df["date"].astype("datetime64[ns]")
df.drop(columns=["Ignore"], inplace=True)


### Data cleaning for fear greed index
df_fear_greed.rename(columns={"timestamp": "date", "value": "fear_greed"}, inplace=True)
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



# Normalize the data using StandardScaler
scaler = MinMaxScaler(feature_range=(0, 1))
split = int(round(len(df) / 100 * 70, 0))
train = df.iloc[:split]
test = df.iloc[split:]

training_set = scaler.fit_transform(train)
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
    X_train, y_train = create_dataset(training_set, look_back, column_index)
    X_test, y_test = create_dataset(test_set, look_back, column_index)

    X_train = np.reshape(X_train, (X_train.shape[0], look_back, X_train.shape[2]))
    X_test = np.reshape(X_test, (X_test.shape[0], look_back, X_test.shape[2]))

    #### Model Creation ####
    tf.keras.utils.set_random_seed(2022)

    model = Sequential()
    model.add(
        Bidirectional(
            LSTM(
                units=32,
                return_sequences=True,
                input_shape=(X_train.shape[1], X_train.shape[2]),
                activation="linear",
            )
        )
    )
    model.add(Dropout(0.25))
    model.add(
        Bidirectional(
            LSTM(
                units=16,
                return_sequences=False,
                input_shape=(X_train.shape[1], X_train.shape[2]),
                activation="linear",
            )
        )
    )
    model.add(Dense(units=1, activation="sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    #### Model Training and Model Evaluation ####
    history = model.fit(
        X_train,
        y_train,
        epochs=20,
        batch_size=70,
        verbose=2,
        shuffle=True,
        validation_split=0.1,
    )

    # Make predictions on the test data
    trainPredict_proba = model.predict(X_train)
    testPredict_proba = model.predict(X_test)

    # Round the predictions to 0 or 1 based on their direction of price movement
    trainPredict = np.round(trainPredict_proba)
    testPredict = np.round(testPredict_proba)

    # Calculate the accuracy of the model
    trainAccuracy = accuracy_score(y_train, trainPredict.flatten())
    testAccuracy = accuracy_score(y_test, testPredict.flatten())

    print(f"Lag is {look_back}")
    print(f"train accuracy: {trainAccuracy}")
    print(f"test accuracy: {testAccuracy}")


train_model_lookback(1)
train_model_lookback(3)
train_model_lookback(5)
