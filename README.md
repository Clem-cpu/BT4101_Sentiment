# BT4101_Sentiment

This project provides a dashboard for users to track the direction of price movement of Bitcoin and its predicted next day movement based on the sentiment analysis of Reddit data and cryptocurrency news articles. Deep learning models are used to incorporate this sentiment analysis and market data such as the Bitcoin Fear and Greed Index into its prediction and hopes to provide users an additional technical indicator to use for their trading strategies


## Folder Structure

### src/cron_jobs

Folder contains all the necessary modules for scheduling the cron job

1. market_data.py
* Module for scraping Bitcoin price and fear and greed index

2. news_website.py
* Module for scraping cryptocurrency news websites

3. reddit_scraping.py
* Module for scraping reddit on a daily basis

4. price_prediction.py
* Module for predicting the price movement using the best CNN 3 day lag model

### src/reddit

This folder contains modules for handling the training reddit data

1. reddit_cleaning.py
* Cleans the reddit training data obtained from Kaggle with historical reddit posts from 2022-01-01 to 2022-12-31 https://www.kaggle.com/datasets/leukipp/reddit-crypto-data 

2. reddit_sentiment_analysis.py
* Perform sentiment analysis on the training dataset


### src/models

This folder contains all the models

1. cnn_price_prediction.py
* CNN price prediction model that tests the 3 different lag periods


2. bilstm_price_prediction.py
* Bidirectional LSTM price prediction model that tests the 3 different lag periods

3. lstm_price_prediction.py
* LSTM price prediction model that tests the 3 different lag periods


### src/news_website

Folder contains modules for handling cryptocurrency news

1. news_sentiment_analysis.py
* Module for conducting sentiment analysis on the news

2. web_scraping.py
* Module for scraping news articles from DailyCoin and CryptoDaily


### src/test_reddit_scraper

Folder contains modules for testing the reddit scraper 

1. news_sentiment_analysis.py
* Cryptocurrency news sentiment analysis

2. reddit_sentiment_analysis.py
* Reddit sentiment analysis for the reddit data scraped from the cron job

3. test_price_prediction.py
* Price prediction test that uses the best 3 day lag CNN model



### src/market_data

Folder contains modules responsible for scraping market data

1. bitcoin_price_scrapping.py
* Bitcoin price scrapping from Binance API

2. fear_greed_index.py
* Fear and greed index scraping from https://alternative.me/crypto/fear-and-greed-index/ 

### src/main.py

Folder which contains the module to be run for the CRON job to scrape data on a daily basis

### test/application_files

Folder contains files necessary for creating the dashboard application

### test/news_articles

Folder contains scraped news articles and the sentiment analysis

### test/reddit

Folder contains training reddit dataset obtained from Kaggle

### test/reddit_scraper

Folder contains reddit scraper files 

### test/save_files

Folder contains saved models and scalers


### templates

Folder contains html files necessary for creating the dashboard

### app.py

Contains the dashboard application of the prediction

### reddit.json

Contains the necessary reddit credentials that users have to input for the cron job to be scheduled 


## Installation

Set up a virtual environment using the requirement.txt file or run pip install -r requirement.txt



## Set up and Configuration

Edit the reddit.json file to contain the necessary credentials from setting up a reddit developer account. More information can be found here https://www.reddit.com/wiki/api/

Schedule a CRON job for the main.py file to occur on a daily basis at 07:59:00 UTC. 

More information on how to set up a CRON job can be found at https://www.hostinger.com/tutorials/cron-job




