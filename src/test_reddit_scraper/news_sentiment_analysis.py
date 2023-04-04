import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

df = pd.read_csv("./test/news_articles/crypto_news_articles.csv")

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(by="date")
df = df.loc[df["date"] >= "2023-02-22"]
df = df.loc[df["date"] <= "2023-03-13"]

tokenizer = AutoTokenizer.from_pretrained(
    "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
)
model = AutoModelForSequenceClassification.from_pretrained(
    "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
)

nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

count = 0
title_labels = []
title_scores = []
text_labels = []
text_scores = []

for index, row in df.copy().iterrows():
    title_output = nlp(row["title"])[0]

    title_labels.append(title_output["label"])
    title_scores.append(title_output["score"])

    text_output = nlp(row["text"])[0]
    text_labels.append(text_output["label"])
    text_scores.append(text_output["score"])


df["title_label"] = title_labels
df["title_score"] = title_scores

df["text_label"] = text_labels
df["text_scores"] = text_scores

## Data cleaning for news sentiment
df = df[["date", "title_label", "title_score", "text_label", "text_scores"]]
df["date"] = df["date"].astype("datetime64[ns]")


def aggregate_data(df):
    # Group by date
    grouped = df.groupby("date")

    # Majority voting for header_label and text_label
    title_label = grouped["title_label"].agg(lambda x: x.value_counts().idxmax())
    text_label = grouped["text_label"].agg(lambda x: x.value_counts().idxmax())

    # Mean of header_score and text_score with final chosen label
    title_score = grouped.apply(
        lambda x: x[x["title_label"] == x["title_label"].iloc[0]]["title_score"].mean()
    )
    text_score = grouped.apply(
        lambda x: x[x["text_label"] == x["text_label"].iloc[0]]["text_scores"].mean()
    )

    # Combine results into a single dataframe
    result = pd.DataFrame(
        {
            "title_label": title_label,
            "title_score": title_score,
            "text_label": text_label,
            "text_scores": text_score,
        }
    )
    result = result.reset_index()

    return result


df = aggregate_data(df)


def one_hot_encode(df):
    # One-hot encode header_label column
    header_encoded = pd.get_dummies(df["title_label"])
    header_encoded.rename(
        columns={
            "negative": "title_negative",
            "neutral": "title_neutral",
            "positive": "title_positive",
        },
        inplace=True,
    )

    # One-hot encode text_label column
    text_encoded = pd.get_dummies(df["text_label"])
    text_encoded.rename(
        columns={
            "negative": "text_negative",
            "neutral": "text_neutral",
            "positive": "text_positive",
        },
        inplace=True,
    )

    # Combine the encoded columns with the original dataframe
    result = pd.concat([df, header_encoded, text_encoded], axis=1)

    return result


df = one_hot_encode(df)
df.drop(columns=["title_label", "text_label"], inplace=True)


print(df.info())
print(df.head(5))
print(df.tail(5))
df.to_csv("./test/news_articles/news_sentiment_test.csv", index=False)
