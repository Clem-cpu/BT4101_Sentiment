import pandas as pd
import numpy as np
import re

keep_cols = ["created", "title", "selftext", "removed"]
names = [
    "cryptocurrency",
    "bitcoin",
    "binance",
    "cryptocurrencytrading",
    "ethtrader",
    "crypto_com",
    "cryptomarkets",
    "crypto_currency_news",
    "coinbase",
    "btc",
    "crypto_general",
]
dfs = []
for name in names:
    temp = pd.read_csv(
        f"./test/reddit/reddit_posts/{name}/submission.csv", usecols=keep_cols
    )
    dfs.append(temp)

df = pd.concat(dfs, ignore_index=True)
df["date"] = df["created"].apply(lambda x: pd.to_datetime(x, unit="s"))
df["date"] = df["date"].dt.date
# Only keep posts which have not been removed
df = df[df["removed"] == 0]

df.drop(columns=["created", "removed"], inplace=True)

# Remove all rows based on first 2 heuristics
df = df[
    df["selftext"].str.contains("give away| giving away| register | pump | join")
    == False
]
df = df[
    df["title"].str.contains("give away| giving away| register | pump | join") == False
]

# Remove rows with more than 14 hashtags
hashtag_pattern = r"#\w+"
df = df[df["selftext"].apply(lambda x: len(re.findall(hashtag_pattern, x))) <= 14]
df = df[df["title"].apply(lambda x: len(re.findall(hashtag_pattern, x))) <= 14]


df["selftext"] = df["selftext"].str[:512]
df["title"] = df["title"].str[:512]

df.to_csv("./test/reddit/cleaned_posts.csv", index=False)

print(df.head(30))
print(df.info())
