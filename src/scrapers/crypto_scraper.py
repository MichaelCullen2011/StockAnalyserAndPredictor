import json
import pandas as pd
import numpy as np
import os
import sys
import requests
import zipfile
from io import BytesIO
from pathlib import Path
from binance.client import Client
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

BASE_DIR = Path(__file__).resolve().parent.parent

from src.gcp.clean_data import clean_scraped_crypto_df

_crypto_data = os.path.join(BASE_DIR, "datasets", "crypto")
with open(
    os.path.join(BASE_DIR, "credentials", "binance-credentials.json")
) as config_file:
    data = json.load(config_file)
binance_key = data["BINANCE_KEY"]
binance_secret = data["BINANCE_SECRET"]

def connect_binance():
    return Client(binance_key, binance_secret)

def binance_historical_data(client, symbol: str, ts: str = "1m", timescale: str = 12):
    print(f"getting {timescale} days of {ts} binance historical data for {symbol}...")
    interval = (
        Client.KLINE_INTERVAL_1DAY if ts == "1d" else Client.KLINE_INTERVAL_1MINUTE
    )
    end_date = datetime.now()
    start_date = datetime.now() - relativedelta(days=timescale) if ts == "1m" else datetime.now() - relativedelta(months=timescale)
    
    # sets to beginning of day at midnight
    end_date = end_date.replace(hour=0, minute=0, second=0, microsecond=0)
    start_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
    
    symbol = "XNO" if symbol == "NANO" else symbol
    klines = client.get_historical_klines(
        symbol=f"{symbol}USDT",
        interval=interval,
        start_str=str(start_date.timestamp() * 1000),
        end_str=str(end_date.timestamp()*1000)
    )
    # unpacks klines and adds to a df
    df = pd.DataFrame(
        np.array(klines).reshape(-1, 12),
        dtype=float,
        columns=(
            "time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_asset_volume",
            "number_of_trades",
            "taker_buy_base_asset_volume",
            "taker_buy_quote_asset_volume",
            "ignore",
        ),
    )
    return clean_scraped_crypto_df(df)

def binance_specific_data(client, symbol: str, scrape_date: str = datetime.now().strftime("%Y-%m-%d")):
    print(f"getting 1m binance historical data for {symbol} on {scrape_date}...")
    interval = Client.KLINE_INTERVAL_1MINUTE

    start_date = datetime.strptime(scrape_date, '%Y-%m-%d')
    end_date = start_date.replace(hour=23, minute=59, second=00, microsecond=0)
    assert end_date < datetime.now()

    symbol = "XNO" if symbol == "NANO" else symbol
    klines = client.get_historical_klines(
        symbol=f"{symbol}USDT",
        interval=interval,
        start_str=str(start_date.timestamp() * 1000),
        end_str=str(end_date.timestamp()*1000)
    )
    # unpacks klines and adds to a df
    df = pd.DataFrame(
        np.array(klines).reshape(-1, 12),
        dtype=float,
        columns=(
            "time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_asset_volume",
            "number_of_trades",
            "taker_buy_base_asset_volume",
            "taker_buy_quote_asset_volume",
            "ignore",
        ),
    )
    return df


def get_data(symbol: str, date: str, ts: str = "monthly") -> pd.DataFrame:
    url = f"https://data.binance.vision/data/spot/{ts}/klines/{symbol}USDT/1m/"
    endpoint = f"{symbol}USDT-1m-{date}"
    url = f"{url}{endpoint}.zip"
    try:
        print(f"fetching {symbol} data for {date}...")
        r = requests.get(url)
        r.raise_for_status()
        zip_file = zipfile.ZipFile(BytesIO(r.content))
        df = pd.read_csv(zip_file.open(f"{endpoint}.csv"))
        with zip_file as zf:
            zf.extractall(os.path.join(_crypto_data, "binance_scraped"), zf.namelist())
        return df
    except requests.exceptions.HTTPError as error:
        print(f"{symbol} data does not exist for {symbol} {date}!")
        return error


def save_local(df: pd.DataFrame, symbol: str) -> None:
    df.to_csv(os.path.join(_crypto_data, f"{symbol}USDT-1m-Total.csv"))


def load_local(symbol: str, date: str) -> pd.DataFrame:
    return pd.read_csv(
        os.path.join(_crypto_data, "binance_scraped", f"{symbol}USDT-1m-{date}.csv")
    )


def check_date_exists(symbol: str, date: str) -> bool:
    return os.path.exists(
        os.path.join(_crypto_data, "binance_scraped", f"{symbol}USDT-1m-{date}.csv")
    )


def run_scraper(symbol: str, dates: list, ts: str = "monthly") -> pd.DataFrame:
    dates = (
        dates.strftime("%Y-%m-%d").tolist()
        if ts == "daily"
        else dates.strftime("%Y-%m").tolist()
    )
    total_df = pd.DataFrame.from_dict(
        {"time": [], "open": [], "high": [], "low": [], "close": [], "volume": []}
    )
    if symbol == "NANO":
        symbol = "XNO"
    for date in dates:
        df = (
            # get_data(symbol, date, ts)
            binance_specific_data(connect_binance(), symbol, date)
            if not check_date_exists(symbol, date)
            else load_local(symbol, date)
        )
        if type(df) == requests.exceptions.HTTPError:
            pass
        else:
            df = clean_scraped_crypto_df(df)
            total_df = pd.concat([total_df, df])

    total_df = total_df.set_index("time")
    print(total_df)
    return total_df


if __name__ == "__main__":
    coins = [
        "BTC",
        "ETH",
        "ADA",
        "NANO",
        "XMR",
        "BNB",
        "SOL",
        "UNI",
        "SHIB",
        "DOGE",
        "DOT",
        "ALGO",
        "LRC",
    ]
    coins = ["BTC"]

    # OLD METHOD - get data for pd date range
    dates = pd.date_range("2021-11-01", "2021-11-03", freq="MS")
    for coin in coins:
        total_df = run_scraper(coin, dates, ts="daily")

    # data = {coin: {} for coin in coins}
    
    # # get 1m historical data from start date to today
    # # for coin in coins:
    # #     data[coin] = binance_historical_data(connect_binance(), coin, "1m", 2)

    # # get 1m historical data for a specific day
    # dates = ["2022-08-23", "2022-02-01", "2020-01-01"]
    # for coin in coins:
    #     for date in dates:
    #         data[coin][date] = binance_specific_data(connect_binance(), coin, date)
    #         # maybe flatten the dict and make it one single df 

    print(total_df)
