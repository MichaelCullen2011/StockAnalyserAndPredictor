import pandas as pd
import os
import sys
import requests
import zipfile
from io import BytesIO
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
# BASE_DIR = os.path.dirname(__file__)

sys.path.append(os.path.join(BASE_DIR, "gcp"))
from clean_data import clean_scraped_crypto_df

_root = BASE_DIR
_crypto_data = os.path.join(_root, "datasets", "crypto")


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
            get_data(symbol, date, ts)
            if not check_date_exists(symbol, date)
            else load_local(symbol, date)
        )
        if type(df) == requests.exceptions.HTTPError:
            pass
        else:
            df = clean_scraped_crypto_df(df)
            # total_df = total_df.append(df)
            total_df = pd.concat([total_df, df])
            # if os.path.exists(os.path.join(_crypto_data, f"{symbol}USDT-1m-Total.csv")):
            #     prev_df = pd.read_csv(
            #         os.path.join(_crypto_data, f"{symbol}USDT-1m-Total.csv")
            #     )
            #     total_df = prev_df.append(total_df)

    total_df = total_df.set_index("time")
    # save_local(total_df, symbol)
    return total_df


if __name__ == "__main__":
    symbols = [
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
    symbols = ["BTC"]
    dates = pd.date_range("2017-08-01", "2021-11-01", freq="MS")
    for symbol in symbols:
        total_df = run_scraper(symbol, dates, ts="monthly")
