import os
from pathlib import Path
import sys

import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(os.path.join(BASE_DIR, "scripts"))

from prediction_scripts.data_models.dir_model import DirVars
from prediction_scripts.data_models.ml_vars_model import MLVars
from prediction_scripts.data_models.scraped_model import ScrapedData
import analysis_scripts.rsi as rsi_calc


class ExtraStockData:
    """stores the stock data class alongside the extra columns stock data"""

    stock_data: ScrapedData
    subsampled_data: dict
    date_time: dict
    last_date: np.datetime64
    extra_cols_dataframe: dict
    column_names: list

    def __init__(self, stock_data: ScrapedData):
        self.stock_data: ScrapedData = stock_data
        self.subsampled_data: dict = {ticker: [] for ticker in self.stock_data.symbols}
        self.date_time: dict = {ticker: [] for ticker in self.stock_data.symbols}
        self.last_date: np.datetime64 = ""
        self.subsample()

        self.extra_cols_dataframe: dict = {
            ticker: [] for ticker in self.stock_data.symbols
        }
        self.column_names: list = []
        self.extra_cols_calculations()

    def subsample(self) -> dict:
        """subsamples the data depending on timeframes (eg. splits into hour only data)"""
        for ticker in self.stock_data.symbols:
            try:
                dataframe = self.stock_data.data_dict[ticker]
            except KeyError:
                dataframe = self.stock_data.data_dict  # just a single df and not a dict
            dataframe = dataframe.reset_index()
            dataframe = dataframe[["time", "close", "open", "high", "low", "volume"]]
            date_time = pd.to_datetime(
                dataframe.pop("time"), format="%Y-%m-%d %H:%M:%S"
            )

            self.last_date = date_time[-1:].values[0]
            self.date_time[ticker] = date_time
            self.subsampled_data[ticker] = dataframe

    def extra_cols_calculations(self) -> dict:
        """calculates additional cols for the dataframes (emas, smas, macd, rsi)"""
        for ticker in self.subsampled_data.keys():
            dataframe = self.subsampled_data[ticker]
            # emas
            emas_used = [3, 5, 10, 12, 26, 30]
            smas_used = [10, 30, 50]
            for x in emas_used:
                ema = x
                dataframe["ema_" + str(ema)] = round(
                    dataframe.iloc[:, 0].ewm(span=ema, adjust=False).mean(), 2
                )
            # smas
            for x in smas_used:
                sma = x
                dataframe["sma_" + str(sma)] = round(
                    dataframe.iloc[:, 0].ewm(span=sma, adjust=False).mean(), 2
                )
            # macd
            dataframe["macd"] = dataframe["ema_12"] - dataframe["ema_26"]
            # rsi
            up_prices, down_prices = rsi_calc.up_down(dataframe["close"].values)
            avg_gain, avg_loss = rsi_calc.averages(up_prices, down_prices)
            RS, RSI = rsi_calc.rsi(dataframe["close"].values, avg_gain, avg_loss)
            dataframe["rsi"] = RSI
            # social media sentiment

            ## add the new dataframe to extra_cols_dataframe
            dataframe = dataframe[
                [
                    "close",
                    "open",
                    "high",
                    "low",
                    "volume",
                    "ema_3",
                    "ema_5",
                    "ema_10",
                    "ema_30",
                    "sma_10",
                    "sma_30",
                    "sma_50",
                    "macd",
                    "rsi",
                ]
            ]
            self.extra_cols_dataframe[ticker] = dataframe
        self.column_names = dataframe.columns


if __name__ == "__main__":
    stock_data = ScrapedData(
        MLVars(timescale="days", data_type="crypto"),
        DirVars(["BTC"]),
        data_type="crypto",
    )
    extra_stock_data = ExtraStockData(stock_data)
    print(extra_stock_data.extra_cols_dataframe)
