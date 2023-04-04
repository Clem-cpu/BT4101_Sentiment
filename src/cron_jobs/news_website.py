import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from bs4 import BeautifulSoup
import requests
import datetime

############################################### CRYPTO DAILY SCRAPING ###############################################


def search_cryptodaily(coin: str, titles, texts, dates) -> pd.DataFrame:
    url = f"https://cryptodaily.co.uk/search?q={coin}"
    request = requests.get(url)
    soup = BeautifulSoup(request.text, "html.parser")

    for news in soup.find_all("a", {"class": "post-item"}):
        article_type = news.find("span", {"class": "hb-tag"}).get_text(strip=True)
        if article_type != "trading":
            title = news.find("h3", {"class": "hb-title"}).get_text(strip=True)
            date = news.find("span", {"class": "hb-date"}).get_text(strip=True)
            paragraph = news.find("div", {"class": "hbs-text"}).p.get_text(strip=True)
            temp = pd.to_datetime(date).date()
            cutoff = datetime.datetime.strptime("2022-12-31", "%Y-%m-%d").date()
            if temp < cutoff:
                continue

            titles.append(title)
            dates.append(date)
            texts.append(paragraph)


############################################### DAILY COIN SCRAPING ###############################################


def search_dailycoin(coin: str, page_number: int, titles, texts, dates) -> pd.DataFrame:
    url = f"https://dailycoin.com/page/{page_number}?s={coin}"
    agent = {
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.182 Safari/537.36"
    }
    request = requests.get(url, headers=agent)
    soup = BeautifulSoup(request.text, "html.parser")

    for news in soup.find_all("div", {"class": "mkd-post-item-inner"}):
        top_level = news.find("div", {"class": "mkd-pt-content-holder"})
        title = top_level.find("h3", {"class": "mkd-pt-title"}).a.get_text(strip=True)
        text = top_level.find("div", {"class": "mkd-post-excerpt"}).p.get_text(
            strip=True
        )
        date = (
            top_level.find("div", {"class": "mkd-pt-meta-section clearfix"})
            .find("div", {"class": "mkd-post-info-date"})
            .span.get_text(strip=True)
        )
        temp = pd.to_datetime(date).date()
        cutoff = datetime.datetime.strptime("2022-12-31", "%Y-%m-%d").date()
        if temp < cutoff:
            continue

        titles.append(title)
        texts.append(text)
        dates.append(date)


def search_dailycoin_first(coin: str, titles, texts, dates) -> pd.DataFrame:
    url = f"https://dailycoin.com/?s={coin}"
    agent = {
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.182 Safari/537.36"
    }
    request = requests.get(url, headers=agent)
    soup = BeautifulSoup(request.text, "html.parser")

    for news in soup.find_all("div", {"class": "mkd-post-item-inner"}):
        top_level = news.find("div", {"class": "mkd-pt-content-holder"})
        title = top_level.find("h3", {"class": "mkd-pt-title"}).a.get_text(strip=True)
        text = top_level.find("div", {"class": "mkd-post-excerpt"}).p.get_text(
            strip=True
        )
        date = (
            top_level.find("div", {"class": "mkd-pt-meta-section clearfix"})
            .find("div", {"class": "mkd-post-info-date"})
            .span.get_text(strip=True)
        )
        temp = pd.to_datetime(date).date()
        cutoff = datetime.datetime.strptime("2022-12-31", "%Y-%m-%d").date()
        if temp < cutoff:
            continue

        titles.append(title)
        texts.append(text)
        dates.append(date)


############################################### CRON JOB ###############################################


def website_cron():
    titles = []
    texts = []
    dates = []
    for i in range(1, 15):
        try:
            search_cryptodaily(f"Bitcoin&page={i}", titles, texts, dates)
            search_cryptodaily(f"BTC&page={i}", titles, texts, dates)
            search_cryptodaily(f"Btc&page={i}", titles, texts, dates)
            search_cryptodaily(f"cryptocurrency&page={i}", titles, texts, dates)
            print(f"Page {i} Scraped")
        except:
            continue
    search_dailycoin_first("Bitcoin", titles, texts, dates)
    search_dailycoin_first("BTC", titles, texts, dates)
    search_dailycoin_first("Btc", titles, texts, dates)
    search_dailycoin_first("cryptocurrency", titles, texts, dates)

    for i in range(2, 15):
        try:
            search_dailycoin("Bitcoin", i, titles, texts, dates)
            search_dailycoin("BTC", i, titles, texts, dates)
            search_dailycoin("Btc", i, titles, texts, dates)
            search_dailycoin("cryptocurrency", i, titles, texts, dates)
            print(f"Page {i} scraped")
        except:
            continue

    df = pd.DataFrame(
        list(zip(titles, texts, dates)), columns=["title", "text", "date"]
    )

    df["date"] = pd.to_datetime(df["date"])
    df = df.loc[df["date"] >= pd.to_datetime("2022-01-01")]
    df = df.sort_values(by="date")

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
    df["date"] = df["date"].astype("datetime64[ns]").dt.date

    def aggregate_data(df):
        # Group by date
        grouped = df.groupby("date")

        # Majority voting for header_label and text_label
        title_label = grouped["title_label"].agg(lambda x: x.value_counts().idxmax())
        text_label = grouped["text_label"].agg(lambda x: x.value_counts().idxmax())

        # Mean of header_score and text_score with final chosen label
        title_score = grouped.apply(
            lambda x: x[x["title_label"] == x["title_label"].iloc[0]][
                "title_score"
            ].mean()
        )
        text_score = grouped.apply(
            lambda x: x[x["text_label"] == x["text_label"].iloc[0]][
                "text_scores"
            ].mean()
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

    df_senti = one_hot_encode(df)
    df_senti.drop(columns=["title_label", "text_label"], inplace=True)
    df_current = pd.read_csv("./test/application_files/news_sentiment.csv")
    df_final = df_current.append(df_senti, ignore_index=True)
    df_final["date"] = df_final["date"].astype("datetime64[ns]").dt.date
    df_final.drop_duplicates(subset=["date"], keep="first", inplace=True)
    df_final.sort_values(by="date", inplace=True)

    df_final.to_csv("./test/application_files/news_sentiment.csv", index=False)


