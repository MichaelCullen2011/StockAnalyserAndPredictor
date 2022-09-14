from datetime import datetime
from dateutil.relativedelta import relativedelta

import numpy as np
import pandas as pd
import yfinance as yf

import matplotlib.pyplot as plt

def get_data(ticker, timescale):
    today = datetime.today()
    start_date = today - relativedelta(days=int(365 * timescale))
    return yf.download(ticker, start=start_date.strftime("%Y-%m-%d"))["Adj Close"]


def calc_perc_change(data):
    perc_change = data.pct_change() * 100
    return perc_change


def get_days_col(data):
    weekdays = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    data = data.reset_index()
    data["Date"] = pd.to_datetime(data["Date"], format="%Y-%m-%d")
    data["Weekday"] = data["Date"].dt.day_name()
    return data


def aggregate(data):
    return data[["Adj Close", "Weekday"]].groupby(["Weekday"]).mean()


def run(tickers, timescale):
    data = {}
    perc_change = {}
    weekday_perc_change = {}
    aggregated_data = {}
    for ticker in tickers:
        data[ticker] = get_data(ticker, timescale)
        perc_change[ticker] = calc_perc_change(data[ticker])
        weekday_perc_change[ticker] = get_days_col(perc_change[ticker])
        aggregated_data[ticker] = aggregate(weekday_perc_change[ticker])
        print(
            f'Aggregated Data for {ticker}: \n {aggregated_data[ticker].sort_values("Adj Close")}'
        )


if __name__ == "__main__":
    tickers = ["GME", "AAPL", "TSLA"]
    run(tickers=tickers, timescale=0.25)
