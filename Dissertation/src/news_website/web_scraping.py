import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import requests
import datetime

titles = []
texts = []
dates = []


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

            titles.append(title)
            dates.append(date)
            texts.append(paragraph)


for i in range(1, 100):
    try:
        search_cryptodaily(f"Bitcoin&page={i}", titles, texts, dates)
        search_cryptodaily(f"BTC&page={i}", titles, texts, dates)
        search_cryptodaily(f"Btc&page={i}", titles, texts, dates)
        search_cryptodaily(f"cryptocurrency&page={i}", titles, texts, dates)
        print(f"Page {i} Scraped")
    except:
        continue


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

        titles.append(title)
        texts.append(text)
        dates.append(date)


search_dailycoin_first("Bitcoin", titles, texts, dates)
search_dailycoin_first("BTC", titles, texts, dates)
search_dailycoin_first("Btc", titles, texts, dates)
search_dailycoin_first("cryptocurrency", titles, texts, dates)

for i in range(2, 100):
    try:
        search_dailycoin("Bitcoin", i, titles, texts, dates)
        search_dailycoin("BTC", i, titles, texts, dates)
        search_dailycoin("Btc", i, titles, texts, dates)
        search_dailycoin("cryptocurrency", i, titles, texts, dates)
        print(f"Page {i} scraped")
    except:
        continue


df = pd.DataFrame(list(zip(titles, texts, dates)), columns=["title", "text", "date"])
df.to_csv("./test/news_articles/crypto_news_articles.csv", index=False)
