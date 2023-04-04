import pandas as pd
import numpy as np
from datetime import datetime
from tensorflow import keras
import tensorflow as tf
from sklearn.metrics import accuracy_score
import joblib
from keras.models import load_model
import pandas as pd
from datetime import datetime, timedelta
import os
import json


def check_and_add_dates(csv_file):
    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Convert "date" column to datetime
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")

    # Create a list of all dates between 2022-01-01 to 2023-06-01
    all_dates = pd.date_range(start="2022-01-01", end="2023-06-01", freq="D")

    # Check if all dates are in the "date" column
    missing_dates = set(all_dates) - set(df["date"])

    # If there are missing dates, add them with "probability" column set to None
    if missing_dates:
        missing_df = pd.DataFrame({"date": list(missing_dates)})
        missing_df["probability"] = np.nan
        df = pd.concat([df, missing_df]).sort_values("date").reset_index(drop=True)

    return df


def predict_cron():
    df = pd.read_csv("./test/application_files/bitcoin_prices.csv")
    df_fear_greed = pd.read_csv("./test/application_files/fear_greed.csv")
    df_reddit = pd.read_csv("./test/application_files/reddit_sentiment.csv")
    df_news = pd.read_csv("./test/application_files/news_sentiment.csv")

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

    column_names = [
        "title_negative",
        "title_neutral",
        "title_positive",
        "text_negative",
        "text_neutral",
        "text_positive",
    ]
    for column in column_names:
        df[column] = df[column].fillna(0)

    df["Change"] = (df["Close"] > df["Close"].shift(1)).astype(int)
    df.to_csv("./test/application_files/prediction_data.csv", index=False)

    scaler = joblib.load("./test/save_files/scaler.gz")
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

    def model_prediction(look_back: int):
        column_index = df.columns.get_loc("Change")
        X_test, y_test = create_dataset(test_set, look_back, column_index)

        X_test = np.reshape(X_test, (X_test.shape[0], look_back, X_test.shape[2]))
        model = load_model(f"./test/save_files/{look_back}_cnn_model")

        # Make predictions on the test data
        testPredict_proba = model.predict(X_test)

        day = []
        for data in X_test:
            temp = datetime.fromtimestamp(
                scaler.inverse_transform(data)[1][5] / 1000
            ).strftime("%Y-%m-%d")
            day.append(temp)

        # Round the predictions to 0 or 1 based on their direction of price movement
        testPredict = np.round(testPredict_proba)
        # Calculate the accuracy of the model
        testAccuracy = accuracy_score(y_test, testPredict.flatten())

        output_dict = {
            "date": day,
            "probability": testPredict_proba.flatten(),
        }
        temp_df = pd.DataFrame.from_dict(output_dict)
        temp_df.to_csv("./test/application_files/output_2.csv", index=False)

        output_df = check_and_add_dates("./test/application_files/output_2.csv")
        output_df.loc[output_df["probability"] <= 0.1, "probability"] = None
        output_df.to_csv("./test/application_files/output.csv", index=False)
        os.remove("./test/application_files/output_2.csv")

        print(f"Lag is {look_back}")
        print(f"test accuracy: {testAccuracy}")
        filepath = "./test/application_files/model_accuracy.json"
        with open(filepath, "w") as file:
            json.dump(round(testAccuracy,4), file)


    model_prediction(3)

