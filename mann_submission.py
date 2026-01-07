import pandas as pd
import numpy as np
import sqlite3
import spacy
from datetime import datetime
import pytz
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import os

print("Loading spaCy model...")
nlp = spacy.load("en_core_web_sm")
print("spaCy is ready!")

print("\nMaking fake news data...")
news_data = {
    "headline": [
        "Apple releases new iPhone and stock goes up",
        "How to bake apple pie at home",
        "Amazon makes big profit this year",
        "Amazon rainforest needs help",
        "Tesla new car is very fast",
        "Microsoft updates Windows",
        "Google search gets better",
        "Apple tree in my garden has many fruits"
    ],
    "company": ["Apple", "Apple", "Amazon", "Amazon", "Tesla", "Microsoft", "Google", "Apple"],
    "timestamp_utc": [
        "2024-01-15 10:00:00",
        "2024-01-15 12:00:00",
        "2024-01-15 14:00:00",
        "2024-01-16 09:00:00",
        "2024-01-16 18:00:00",
        "2024-01-17 11:00:00",
        "2024-01-18 15:00:00",
        "2024-01-19 08:00:00"
    ],
    "sentiment_score": [0.8, -0.2, 0.9, 0.0, 0.7, 0.6, 0.5, -0.1]
}

df_news = pd.DataFrame(news_data)
df_news['timestamp_utc'] = pd.to_datetime(df_news['timestamp_utc']).dt.tz_localize('UTC')
print("News data created:")
print(df_news)

print("\nFiltering news with NER...")
good_news = []
for i in range(len(df_news)):
    row = df_news.iloc[i]
    doc = nlp(row["headline"])
    found_org = False
    for ent in doc.ents:
        if row["company"].lower() in ent.text.lower() and ent.label_ == "ORG":
            found_org = True
            break
    if found_org:
        good_news.append(row)

df_news_clean = pd.DataFrame(good_news).reset_index(drop=True)
print("Clean news (only real companies):")
print(df_news_clean)

print("\nConverting to Indian time and trading day...")
IST = pytz.timezone('Asia/Kolkata')
market_close_time = datetime.strptime("15:30", "%H:%M").time()

trading_days = []
for ts in df_news_clean["timestamp_utc"]:
    ts_ist = ts.astimezone(IST)
    if ts_ist.weekday() >= 5:
        days_to_add = 7 - ts_ist.weekday()
        new_day = ts_ist + pd.Timedelta(days=days_to_add)
        trading_day = new_day.date()
    elif ts_ist.time() > market_close_time:
        trading_day = (ts_ist + pd.Timedelta(days=1)).date()
        if trading_day.weekday() >= 5:
            trading_day = trading_day + pd.Timedelta(days=7 - trading_day.weekday())
    else:
        trading_day = ts_ist.date()
    trading_days.append(trading_day)

df_news_clean["trading_day"] = trading_days
print("News with trading day:")
print(df_news_clean[["headline", "timestamp_utc", "trading_day"]])

print("\nMaking fake stock prices...")
dates = pd.date_range("2024-01-15", "2024-01-22", freq="B")
stock_data = {
    "date": dates,
    "close_price": [150, 152, 155, 153, 158, 160, 159, 162]
}
df_stock = pd.DataFrame(stock_data)
print("Stock prices:")
print(df_stock)

print("\nGrouping news and merging with stock...")
news_by_day = df_news_clean.groupby("trading_day")["sentiment_score"].mean().reset_index()
news_by_day.rename(columns={"trading_day": "date", "sentiment_score": "avg_sentiment"}, inplace=True)

df_final = pd.merge(df_stock, news_by_day, on="date", how="left")
df_final["avg_sentiment"] = df_final["avg_sentiment"].fillna(0)
df_final["news_count"] = df_news_clean.groupby("trading_day").size().reindex(df_final["date"]).fillna(0).values
print("Final table:")
print(df_final)

print("\nTesting if prices are stationary...")
prices = df_final["close_price"]

result = adfuller(prices)
print("Raw prices p-value:", result[1])
if result[1] > 0.05:
    print("Raw prices are NOT stationary")

returns = prices.diff().dropna()
result2 = adfuller(returns)
print("Returns p-value:", result2[1])
if result2[1] < 0.05:
    print("Returns ARE stationary! Good for models.")

print("\nSaving to database...")
conn = sqlite3.connect("my_project.db")
df_final.to_sql("stock_with_sentiment", conn, if_exists="replace")
conn.close()
print("Saved to my_project.db")

print("\nMaking a plot...")
plt.figure(figsize=(10, 5))
plt.plot(df_final["date"], df_final["close_price"], label="Stock Price", marker="o")
plt.plot(df_final["date"], df_final["avg_sentiment"]*50 + 140, label="Sentiment (scaled)", marker="x")
plt.xlabel("Date")
plt.ylabel("Price / Scaled Sentiment")
plt.title("Stock Price and News Sentiment")
plt.legend()
plt.grid()
plt.show()

print("\nSome NumPy calculations...")
price_array = np.array(df_final["close_price"])
print("Average price:", np.mean(price_array))
print("Highest price:", np.max(price_array))
print("Lowest price:", np.min(price_array))