import datetime
import pandas as pd
import praw
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
import os
import json
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.tokenize import TweetTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
import nltk
import re


def upload_file(filename):
    SCOPES = ["https://www.googleapis.com/auth/drive"]
    SERVICE_ACCOUNT_FILE = "service.json"
    credentials = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES
    )
    folder_id = "1nxWszlcsqnZEwEjG_IcPK-uk2wtngIP0"
    basename = os.path.basename(filename).split(".")[0]

    with build("drive", "v3", credentials=credentials) as service:
        # Call the Drive v3 API
        media = MediaFileUpload(filename)
        body = {
            "name": f"{basename}.csv",
            "mimetype": "application/vnd.google-apps.file",
            "parents": [folder_id],
        }

        service.files().create(body=body, media_body=media).execute()


def cleaning(data):
    # remove urls
    lem = WordNetLemmatizer()
    stop_words = nltk.corpus.stopwords.words(["english"])
    tweet_without_url = re.sub(r"http\S+", " ", data)

    # remove hashtags
    tweet_without_hashtag = re.sub(r"#\w+", " ", tweet_without_url)

    # 3. Remove mentions and characters that not in the English alphabets
    tweet_without_mentions = re.sub(r"@\w+", " ", tweet_without_hashtag)
    precleaned_tweet = re.sub("[^A-Za-z]+", " ", tweet_without_mentions)

    # 2. Tokenize
    reddit_tokens = TweetTokenizer().tokenize(precleaned_tweet)
    # tweet_tokens =

    # 3. Remove Puncs
    tokens_without_punc = [w for w in reddit_tokens if w.isalpha()]

    # 4. Removing Stopwords
    tokens_without_sw = [t for t in tokens_without_punc if t not in stop_words]
    # 5. lemma
    text_cleaned = [lem.lemmatize(t) for t in tokens_without_sw]

    # 6. Joining
    return " ".join(text_cleaned)


def aggregate_data(df):
    # Group by date
    grouped = df.groupby("date")

    # Majority voting for header_label and text_label
    reddit_neg = grouped["reddit_neg"].mean()
    reddit_neu = grouped["reddit_neu"].mean()
    reddit_pos = grouped["reddit_pos"].mean()
    reddit_comp = grouped["reddit_comp"].mean()

    reddit_neg_2 = grouped["reddit_neg_2"].mean()
    reddit_neu_2 = grouped["reddit_neu_2"].mean()
    reddit_pos_2 = grouped["reddit_pos_2"].mean()
    reddit_comp_2 = grouped["reddit_comp_2"].mean()
    # Combine results into a single dataframe
    result = pd.DataFrame(
        {
            "reddit_neg": reddit_neg,
            "reddit_neu": reddit_neu,
            "reddit_pos": reddit_pos,
            "reddit_comp": reddit_comp,
            "reddit_neg_2": reddit_neg_2,
            "reddit_neu_2": reddit_neu_2,
            "reddit_pos_2": reddit_pos_2,
            "reddit_comp_2": reddit_comp_2,
        }
    )
    result = result.reset_index()

    return result


def sentiment_analysis(filename, today_date):
    df = pd.read_csv(filename)
    df = df[df["selftext"].str.contains("https | http | giveaway") == False]
    df["selftext"] = df["selftext"].str[:512]
    df["title"] = df["title"].str[:512]
    df = df[df["title"].str.contains("giveaway") == False]

    df["title"] = df["title"].astype(str)
    df["selftext"] = df["selftext"].astype(str)
    df["title"] = df["title"].apply(cleaning)
    df["selftext"] = df["selftext"].apply(cleaning)

    sid_obj = SentimentIntensityAnalyzer()

    count = 0
    negative = []
    neutral = []
    positive = []
    comp = []

    negative_2 = []
    neutral_2 = []
    positive_2 = []
    comp_2 = []

    for index, row in df.copy().iterrows():
        title = row["title"]
        sentiment_dict = sid_obj.polarity_scores(title)
        neg_score = sentiment_dict["neg"]
        neu_score = sentiment_dict["neu"]
        pos_score = sentiment_dict["pos"]
        comp_score = sentiment_dict["compound"]

        negative.append(neg_score)
        neutral.append(neu_score)
        positive.append(pos_score)
        comp.append(comp_score)

        text = row["selftext"]
        sentiment_dict = sid_obj.polarity_scores(text)
        neg_score = sentiment_dict["neg"]
        neu_score = sentiment_dict["neu"]
        pos_score = sentiment_dict["pos"]
        comp_score = sentiment_dict["compound"]

        negative_2.append(neg_score)
        neutral_2.append(neu_score)
        positive_2.append(pos_score)
        comp_2.append(comp_score)

    df["reddit_neg"] = negative
    df["reddit_neu"] = neutral
    df["reddit_pos"] = positive
    df["reddit_comp"] = comp

    df["reddit_neg_2"] = negative_2
    df["reddit_neu_2"] = neutral_2
    df["reddit_pos_2"] = positive_2
    df["reddit_comp_2"] = comp_2

    ## Data cleaning for news sentiment
    df = df[
        [
            "date",
            "reddit_neg",
            "reddit_neu",
            "reddit_pos",
            "reddit_comp",
            "reddit_neg_2",
            "reddit_neu_2",
            "reddit_pos_2",
            "reddit_comp_2",
        ]
    ]
    df["date"] = df["date"].astype("datetime64[ns]")
    df["date"] = df["date"].dt.date
    df = aggregate_data(df)
    df.to_csv(f"./{today_date}_reddit_sentiment.csv", index=False)


def lambda_handler(event=None, context=None):
    names = [
        "Cryptocurrency",
        "Bitcoin",
        "binance",
        "CryptoCurrencyTrading",
        "CryptoMarkets",
        "ethtrader",
        "btc",
        "Crypto_Currency_News",
        "crypto_com",
        "coinbase",
        "crypto_general",
    ]

    with open("reddit.json") as json_file:
        reddit_json = json.load(json_file)
        reddit = praw.Reddit(
            client_id=reddit_json["client_id"],
            client_secret=reddit_json["client_secret"],
            user_agent=reddit_json["user_agent"],
        )

        post_data = []
        for name in names:
            subreddit = reddit.subreddit(name)
            top_posts = subreddit.top(time_filter="day", limit=500)

            # Create a list to store the post data
            # Loop through the top posts and store the date, title, and selftext in the post_data list
            for post in top_posts:
                post_data.append([post.created_utc, post.title, post.selftext])

        df = pd.DataFrame(post_data, columns=["date", "title", "selftext"])

        # Convert the "date" column to a pandas datetime object
        df["date"] = pd.to_datetime(df["date"], unit="s").dt.date
        today_date = datetime.datetime.now().date()

        df.to_csv(f"./{today_date}_reddit.csv", index=False)

        upload_file(f"./{today_date}_reddit.csv")

        sentiment_analysis(f"./{today_date}_reddit.csv", today_date)

        upload_file(f"./{today_date}_reddit_sentiment.csv")

        os.remove(f"./{today_date}_reddit.csv")
        os.remove(f"./{today_date}_reddit_sentiment.csv")


