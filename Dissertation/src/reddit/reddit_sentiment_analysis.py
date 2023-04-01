import pandas as pd
import numpy as np
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


df = pd.read_csv("./test/reddit/cleaned_posts.csv")
df["title"] = df["title"].astype(str)
df["selftext"] = df["selftext"].astype(str) 
df["title"] = df["title"].apply(cleaning)
df["selftext"] = df["selftext"].apply(cleaning)

print("Cleaning Done")

sid_obj = SentimentIntensityAnalyzer()


count = 0
title_negative = []
title_neutral = []
title_positive = []
title_comp = []

selftext_negative = []
selftext_neutral = []
selftext_positive = []
selftext_comp = []

loop = 1

for index, row in df.copy().iterrows():
    if count > loop * 40000:
        loop += 1
        print(count)

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

    count += 1

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

print(df.info())
print(df.head(50))
df.to_csv("./test/reddit/reddit_sentiment.csv", index=False)
