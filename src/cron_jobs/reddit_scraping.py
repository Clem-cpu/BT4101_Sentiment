import datetime
import pandas as pd
import praw
import os
import json
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from tokenizer import tokenizer
from nltk.stem.wordnet import WordNetLemmatizer
import nltk
import re


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
    reddit_tokens = tokenizer.RedditTokenizer().tokenize(precleaned_tweet)
    # tweet_tokens =

    # 3. Remove Puncs
    tokens_without_punc = [w for w in reddit_tokens if w.isalpha()]

    # 4. Removing Stopwords
    tokens_without_sw = [t for t in tokens_without_punc if t not in stop_words]
    # 5. lemma
    text_cleaned = [lem.lemmatize(t) for t in tokens_without_sw]

    # 6. Joining
    return " ".join(text_cleaned)


def sentiment_analysis(df, today_date):
    # Remove all rows based on first 2 heuristics
    df = df[
        df["selftext"].str.contains("give away| giving away| register | pump | join")
        == False
    ]
    df = df[
        df["title"].str.contains("give away| giving away| register | pump | join")
        == False
    ]

    # Remove rows with more than 14 hashtags
    hashtag_pattern = r"#\w+"
    df = df[df["selftext"].apply(lambda x: len(re.findall(hashtag_pattern, x))) <= 14]
    df = df[df["title"].apply(lambda x: len(re.findall(hashtag_pattern, x))) <= 14]

    df["selftext"] = df["selftext"].str[:512]
    df["title"] = df["title"].str[:512]
    df["title"] = df["title"].astype(str)
    df["selftext"] = df["selftext"].astype(str)
    df["title"] = df["title"].apply(cleaning)
    df["selftext"] = df["selftext"].apply(cleaning)

    sid_obj = SentimentIntensityAnalyzer()

    title_negative = []
    title_neutral = []
    title_positive = []
    title_comp = []

    selftext_negative = []
    selftext_neutral = []
    selftext_positive = []
    selftext_comp = []

    for index, row in df.copy().iterrows():
        title = row["title"]
        title_sentiment_dict = sid_obj.polarity_scores(title)
        title_neg_score = title_sentiment_dict["neg"]
        title_neu_score = title_sentiment_dict["neu"]
        title_pos_score = title_sentiment_dict["pos"]
        title_comp_score = title_sentiment_dict["compound"]

        title_negative.append(title_neg_score)
        title_neutral.append(title_neu_score)
        title_positive.append(title_pos_score)
        title_comp.append(title_comp_score)

        selftext = row["selftext"]
        selftext_sentiment_dict = sid_obj.polarity_scores(selftext)
        selftext_neg_score = selftext_sentiment_dict["neg"]
        selftext_neu_score = selftext_sentiment_dict["neu"]
        selftext_pos_score = selftext_sentiment_dict["pos"]
        selftext_comp_score = selftext_sentiment_dict["compound"]

        selftext_negative.append(selftext_neg_score)
        selftext_neutral.append(selftext_neu_score)
        selftext_positive.append(selftext_pos_score)
        selftext_comp.append(selftext_comp_score)

    df["reddit_title_neg"] = title_negative
    df["reddit_title_neu"] = title_neutral
    df["reddit_title_pos"] = title_positive
    df["reddit_title_comp"] = title_comp

    df["reddit_selftext_neg"] = selftext_negative
    df["reddit_selftext_neu"] = selftext_neutral
    df["reddit_selftext_pos"] = selftext_positive
    df["reddit_selftext_comp"] = selftext_comp

    ## Data cleaning for news sentiment
    df = df[
        [
            "date",
            "reddit_title_neg",
            "reddit_title_neu",
            "reddit_title_pos",
            "reddit_title_comp",
            "reddit_selftext_neg",
            "reddit_selftext_neu",
            "reddit_selftext_pos",
            "reddit_selftext_comp",
        ]
    ]
    df["date"] = df["date"].astype("datetime64[ns]")
    df["date"] = df["date"].dt.date

    def aggregate_data(df):
        # Group by date
        grouped = df.groupby("date")

        # Aggregate scores based on the mean of each day
        reddit_title_neg = grouped["reddit_title_neg"].mean()
        reddit_title_neu = grouped["reddit_title_neu"].mean()
        reddit_title_pos = grouped["reddit_title_pos"].mean()
        reddit_title_comp = grouped["reddit_title_comp"].mean()

        reddit_selftext_neg = grouped["reddit_selftext_neg"].mean()
        reddit_selftext_neu = grouped["reddit_selftext_neu"].mean()
        reddit_selftext_pos = grouped["reddit_selftext_pos"].mean()
        reddit_selftext_comp = grouped["reddit_selftext_comp"].mean()

        # Combine results into a single dataframe
        result = pd.DataFrame(
            {
                "reddit_title_neg": reddit_title_neg,
                "reddit_title_neu": reddit_title_neu,
                "reddit_title_pos": reddit_title_pos,
                "reddit_title_comp": reddit_title_comp,
                "reddit_selftext_neg": reddit_selftext_neg,
                "reddit_selftext_neu": reddit_selftext_neu,
                "reddit_selftext_pos": reddit_selftext_pos,
                "reddit_selftext_comp": reddit_selftext_comp,
            }
        )
        result = result.reset_index()

        return result

    df = aggregate_data(df)

    return df


def reddit_cron():
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

        df_senti = sentiment_analysis(df, today_date)
        df_current = pd.read_csv("./test/application_files/reddit_sentiment.csv")

        df_final = df_current.append(df_senti, ignore_index=True)
        df_final.drop_duplicates(subset=['date'], keep='last', inplace=True)
        
        df_final.to_csv(f"./test/application_files/reddit_sentiment.csv", index=False)
