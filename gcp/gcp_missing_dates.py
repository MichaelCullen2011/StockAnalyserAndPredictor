import os
import sys

import pandas as pd
import matplotlib.pyplot as plt
import pandas_market_calendars as mcal
from google.cloud import bigquery

from datetime import datetime, timedelta
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
# BASE_DIR = os.path.dirname(__file__)

sys.path.append(os.path.join(BASE_DIR, "gcp"))
from scrapers.stock_scraper import fetch_data as stock_scraper

# import scrapers.av_scraper as av_scraper
from scrapers.crypto_scraper import run_scraper as crypto_scraper

from clean_data import clean_df, volume_as_int
from get_gcp import connect, get_data, save_gcp

dataset_dir = os.path.join(BASE_DIR, "datasets")
scraped_dir = os.path.join(dataset_dir, "scraped")
gcp_dir = os.path.join(dataset_dir, "gcp")


def plot_data(df: pd.DataFrame, symbol: str) -> None:
    df.plot()
    plt.title(f"{symbol}")


def shift_days(missing_str: list) -> list:
    """AV scrapes from the 3rd month_a to 3rd month_b meaning some days wont be picked up unless we shift"""
    return [
        str(pd.to_datetime(date) - pd.Timedelta(timedelta(days=3)))
        for date in missing_str
    ]


# currently broken due to numerous ways of saving datasets
def fix_missing_dates(
    symbol: str, mcal_days: pd.DatetimeIndex, day_df: pd.DataFrame
) -> None:
    """left joins the trading days to the actual df to see which days have nans and are therefore missing"""
    # if df is from gcp saved df
    mcal_days = pd.DataFrame.from_dict({"time": pd.to_datetime(pd.Series(mcal_days))})

    # does not work currently due to day_df having hours:mins:seconds that arent 0
    big_df = mcal_days.merge(day_df, on="time", how="left")
    # get the missing days by finding which days have NaN values
    missing_days = big_df[big_df.isna().any(axis=1)].reset_index()

    # missing_str = missing_days["time"].astype(str).to_list()
    # print("Number of days to scrape: ", len(missing_str))
    # print("Days to scrape: ", missing_str)

    # # ONLY DO THIS IF SCRAPING FROM AV OR WILL GET SHIFTED DAYS
    # missing_str = shift_days(missing_str)

    # formatted_str = [day.replace("-", "") for day in missing_str]
    # formatted_str = [day.replace(" 00:00:00+00:00", "") for day in formatted_str]
    # formatted_months = list(set([day[:-2] for day in formatted_str]))
    # print("Months to scrape from AV: ", formatted_months)

    # """ functions that grabs the data for the missing days, scrapes for these days and sends the scraped df back """
    # # AV Scraping (getting chunk months but limited to 5 calls a minute)
    # missing_df = av_scraper.run(symbol, formatted_months)
    # missing_df["time"] = pd.to_datetime(missing_df["time"], utc=True)
    # local_gcp_df = clean_data(get_local_gcp_data(symbol))
    # local_gcp_df = local_gcp_df.append(clean_data(missing_df.reset_index()))[
    #     ["open", "high", "low", "close", "volume"]
    # ]
    # print("New updating df: \n", local_gcp_df)
    # local_gcp_df.to_csv(os.path.join(gcp_dir, f"{symbol}-gcp-data.csv"))

    # # IEX Scraping (gets specific day scrapes but runs from monthly credits)
    # missing_df = stock_scraper.fetch_data(symbol, formatted_str)[1]
    # missing_df = convert_time_utc(missing_df)
    # local_gcp_df = clean_data(get_local_gcp_data(symbol))
    # local_gcp_df = local_gcp_df.append(clean_data(missing_df.reset_index()))[["open", "high", "low", "close", "volume"]]
    # local_gcp_df.to_csv(os.path.join(gcp_dir, f"{symbol}-gcp-data.csv"))


def fix_erroneous_dates(
    symbol: str, df: pd.DataFrame, data_type: str = "stocks"
) -> pd.DataFrame:
    """compares the lengths of our day df to the number of actual trading days in this period"""
    day_df = df.copy().reset_index()

    """checks to see if dates are missing in the middle of the data (not new days)"""
    df_start_date, df_end_date = day_df["time"].iloc[0], day_df["time"].iloc[-1]
    inbetween_mcal_days = mcal.get_calendar("NYSE").valid_days(
        start_date=df_start_date, end_date=df_end_date
    )
    all_mcal_days = (
        mcal.get_calendar("NYSE").valid_days(
            start_date=df_end_date.strftime("%Y-%m-%d"),
            end_date=datetime.today().strftime("%Y-%m-%d"),
        )[1:]
        if data_type == "stocks"
        else pd.date_range(
            start=df_end_date.strftime("%Y-%m-%d"),
            end=datetime.today().strftime("%Y-%m-%d"),
        )[1:-1]
    )
    scrape_today = (
        datetime.today().strftime("%Y-%m-%d") == all_mcal_days[-1].strftime("%Y-%m-%d")
        if len(all_mcal_days) > 0
        else False
    )
    all_mcal_days = (
        all_mcal_days[:-1] if scrape_today and len(all_mcal_days) > 0 else all_mcal_days
    )
    if len(all_mcal_days) > 0:
        print(f"Last {symbol} date: {df_end_date}")
        print(f"{len(all_mcal_days)} days to scrape for {symbol}!")
        missing_df = (
            crypto_scraper(symbol=symbol, dates=all_mcal_days, ts="daily")
            if data_type == "crypto"
            else clean_df(stock_scraper(symbol=symbol, dates=all_mcal_days))
        )
        save_gcp(df=missing_df, symbol=symbol, data_type=data_type)
    else:
        print("Upto date!")


def run(client: bigquery.Client, symbol: str, data_type: str = "stocks"):
    print(f"Finding missing dates for {symbol}:")

    # save_gcp_locally(symbol)
    """ load and look at gcp data """
    df = clean_df(get_data(client, symbol, data_type=data_type, ts="days"))

    df = fix_erroneous_dates(symbol, df, data_type=data_type)

    # plot_data(df["close"], symbol)


if __name__ == "__main__":
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
    symbols = ["AAPL"]
    client = connect()
    for symbol in symbols:
        run(client, symbol, data_type="stocks")
    plt.show()
