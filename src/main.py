import pandas as pd
from cron_jobs.news_website import website_cron
from cron_jobs.reddit_scraping import reddit_cron
from cron_jobs.market_data import market_cron
from cron_jobs.price_prediction import predict_cron

if __name__ == "__main__":
    website_cron()
    reddit_cron()
    market_cron()
    predict_cron()
