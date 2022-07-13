import os
from pathlib import Path
import csv
import requests
import json
import time

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent.parent
# BASE_DIR = os.path.dirname(__file__)

API_URL = "https://www.alphavantage.co/query?"
with open(os.path.join(BASE_DIR, "config.json")) as config_file:
    data = json.load(config_file)
API_TOKEN = data["AV_TOKEN"]  # test token begins with T

dataset_dir = os.path.join(BASE_DIR, "datasets")
scraped_dir = os.path.join(dataset_dir, "scraped")
av_dir = os.path.join(dataset_dir, "av")


def fetch_data(symbol: str, dates_to_fetch: list) -> pd.DataFrame:
    with open(os.path.join(av_dir, f"{symbol}-av.csv"), "a") as f:
        write = csv.writer(f)
        write.writerow(["time", "open", "high", "low", "close", "volume"])

    for fetch_date in dates_to_fetch:
        URL = f"{API_URL}function=TIME_SERIES_INTRADAY_EXTENDED&symbol={symbol}&interval=1min&slice={fetch_date}&apikey={API_TOKEN}"
        with requests.Session() as s:
            download = s.get(URL)
            decoded_content = download.content.decode("utf-8")
            cr = csv.reader(decoded_content.splitlines(), delimiter=",")
            rows = list(cr)
            rows = rows[1:]

            with open(os.path.join(av_dir, f"{symbol}-av.csv"), "a") as f:
                write = csv.writer(f)
                write.writerows(rows)

        # only allows for 5 calls per minute
        time.sleep(12)

    df = pd.read_csv(os.path.join(av_dir, f"{symbol}-av.csv"))
    df = df.set_index("time")
    df = df.drop_duplicates()
    df = df.reset_index()
    # os.remove(os.path.join(av_dir, f"{symbol}-av.csv"))
    return df


def run(symbol: str, dates: list) -> pd.DataFrame:
    dates_dict = {
        "202111": "year1month1",
        "202110": "year1month2",
        "202109": "year1month3",
        "202108": "year1month4",
        "202107": "year1month5",
        "202106": "year1month6",
        "202105": "year1month7",
        "202104": "year1month8",
        "202103": "year1month9",
        "202102": "year1month10",
        "202101": "year1month11",
        "202012": "year1month12",
        "202011": "year2month1",
        "202010": "year2month2",
        "202009": "year2month3",
        "202008": "year2month4",
        "202007": "year2month5",
        "202006": "year2month6",
        "202005": "year2month7",
        "202004": "year2month8",
        "202003": "year2month9",
        "202002": "year2month10",
        "202001": "year2month11",
        "201912": "year2month12",
    }
    if dates == "All":
        dates_to_fetch = dates_dict.values()
    else:
        dates_to_fetch = [dates_dict[date] for date in dates]
    print(dates_to_fetch)
    return fetch_data(symbol, dates_to_fetch)


if __name__ == "__main__":
    # 'AAPL', 'TSLA', 'GME', 'ABNB', 'PLTR', 'ETSY', 'ENPH', 'GOOG', 'AMZN', 'IBM', 'DIA', 'IVV', 'NIO'
    symbols = [
        "AAPL",
        "TSLA",
        "GME",
        "ABNB",
        "PLTR",
        "ETSY",
        "ENPH",
        "GOOG",
        "AMZN",
        "IBM",
        "DIA",
        "IVV",
        "NIO",
    ]
    symbols = ["GOOG"]

    coins = [
        "BTC",
        "ETH",
        "BNB",
        "SOL",
        "UNI",
        "SHIB",
        "ADA",
        "NANO",
        "XMR",
        "DOGE",
        "DOT",
        "ALGO",
    ]
    coins = ["BTC"]

    dates = "All"
    for symbol in symbols:
        run(symbol, dates)
