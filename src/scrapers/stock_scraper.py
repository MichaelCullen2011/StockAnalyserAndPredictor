import os
from pathlib import Path
import sys
import requests
import json

import pandas as pd
import pandas_market_calendars as mcal
from google.cloud import bigquery

from datetime import datetime, timedelta

BASE_DIR = Path(__file__).resolve().parent.parent
from src.gcp.get_gcp import connect, get_last_gcp_date, save_gcp


API_URL = "https://cloud.iexapis.com/"
with open(os.path.join(BASE_DIR, "credentials", "config.json")) as config_file:
    data = json.load(config_file)
API_TOKEN = data["IEX_TOKEN"]  # test token begins with T

dataset_dir = os.path.join(BASE_DIR, "datasets")
scraped_dir = os.path.join(dataset_dir, "scraped")


def path_check():
    return os.path.join(BASE_DIR, "config.json")


def fetch_data(symbol: str, dates: list) -> pd.DataFrame:
    frames = {}
    for date in dates:
        date = date.strftime("%Y%m%d")
        print(f"Fetching {symbol} data for {date}...")
        URL_ENDPOINT = (
            f"{API_URL}stable/stock/{symbol}/chart/date/{date}?token={API_TOKEN}"
        )
        resp = requests.get(URL_ENDPOINT)
        resp.raise_for_status()
        df = pd.DataFrame(resp.json())
        if not df.empty:
            df["time"] = df["date"] + " " + df["minute"] + ":00"
            df = df[["time", "open", "high", "low", "close", "volume"]]
            frames[date] = df

    total_df = pd.concat(frames.values()).set_index(keys="time")
    return total_df


# def latest_data(symbol: str, last_30_days: list, client: bigquery.Client):
#     av_scrape_bool = False
#     try:
#         latest_date_entry = get_last_gcp_date(symbol, client)
#         latest_date_entry = pd.to_datetime(latest_date_entry).strftime("%Y%m%d")
#         try:
#             dates_to_fetch = last_30_days[last_30_days.index(latest_date_entry) + 1 :]
#         except ValueError:  # if the last recorded date is not in the last 30 days
#             dates_to_fetch = last_30_days
#     except FileNotFoundError:
#         dates_to_fetch = []  # if cannot find the file then there is NO data
#         av_scrape_bool = True
#     print(dates_to_fetch)
#     return dates_to_fetch[:-1], av_scrape_bool


# def run(symbols: str) -> None:
#     # connect to gcp server
#     client = connect()
#     nyse = mcal.get_calendar("NYSE")
#     last_30_days = (
#         nyse.schedule(
#             start_date=datetime.now() - timedelta(days=30), end_date=datetime.now()
#         )
#         .index.strftime("%Y%m%d")
#         .tolist()
#     )
#     combined_dfs = {}
#     for symbol in symbols:
#         """checks whether fetching data is necessary and if so which dates to fetch"""
#         dates_to_fetch, av_scrape_bool = latest_data(symbol, last_30_days, client)
#         if len(dates_to_fetch) > 0 and not av_scrape_bool:
#             print(
#                 f"Fetching data for {symbol} between {dates_to_fetch[0]} - {dates_to_fetch[-1]} ..."
#             )
#             """get data and store in dictionary so it's easy to access each symbol and a corresponding day"""
#             combined_dfs[symbol] = fetch_data(symbol, dates_to_fetch)
#             """save data as a csv"""
#             print(combined_dfs[symbol])
#             # save_gcp(combined_dfs[symbol], symbol, data_type="stocks")
#         elif av_scrape_bool:
#             print(f"No {symbol} stock data in gcp server!")
#         else:
#             print(f"Data for {symbol} is up to date!")


if __name__ == "__main__":
    print(BASE_DIR)
