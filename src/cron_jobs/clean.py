import pandas as pd
from bs4 import BeautifulSoup
import requests
import time
import datetime

df_current = pd.read_csv("./test/news_articles/news_sentiment.csv")
df_test = pd.read_csv("./test/news_articles/news_sentiment_test.csv")


df_output = df_current.append(df_test)

df_output.to_csv("./test/application_files/news_sentiment.csv", index=False)