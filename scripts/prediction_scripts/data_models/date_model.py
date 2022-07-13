import os
from pathlib import Path
import sys

from datetime import timedelta

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(os.path.join(BASE_DIR, "scripts", "prediction_scripts"))
from data_models.ml_vars_model import MLVars
from data_models.dir_model import DirVars
from data_models.scraped_model import ScrapedData
from data_models.extra_stock_model import ExtraStockData


class Dates:
    """class for storing dates and valid trading days"""

    future: str
    timescale: int
    num_predictions: int
    last_date: np.datetime64
    all_trading_days: pd.DatetimeIndex

    def __init__(self, ML: MLVars, extra_data: ExtraStockData):
        self.future: str = ML.future
        self.timescale: int = ML.timescale
        self.num_predictions = ML.num_predictions
        self.data_type = ML.data_type

        self.last_date: np.datetime64 = extra_data.last_date
        self.future_trading_days: pd.DatetimeIndex = self.get_future_trading_days()

    def get_future_trading_days(self) -> pd.DatetimeIndex:
        """gets the next [future] of trading days and appends to past 2 years"""
        if self.timescale == "days":
            # next future trading days
            if self.data_type == "stocks":
                future_dates = mcal.get_calendar("NYSE").valid_days(
                    start_date=self.last_date,
                    end_date=self.last_date
                    + pd.Timedelta(timedelta(days=self.future + 7)),
                )[1 : 1 + self.future]
            else:
                future_dates = pd.date_range(
                    start=self.last_date,
                    end=self.last_date + pd.Timedelta(timedelta(days=self.future + 7)),
                )[1 : 1 + self.future]
        elif self.timescale == "mins":
            # next future trading mins
            if self.data_type == "stocks":
                future_schedule = mcal.get_calendar("NYSE").schedule(
                    start_date=self.last_date,
                    end_date=self.last_date
                    + pd.Timedelta(timedelta(days=self.future + 7)),
                )[1 : 1 + self.future]
                future_dates = mcal.date_range(future_schedule, frequency="1Min")[
                    : self.num_predictions
                ]
            else:
                future_dates = pd.date_range(
                    start=self.last_date,
                    end=self.last_date + pd.Timedelta(timedelta(days=self.future)),
                    freq="T",
                    tz="UTC",
                )[1 : self.num_predictions + 1]
        return future_dates


if __name__ == "__main__":
    ML_vars = MLVars(timescale="mins", future=1, data_type="crypto")
    stock_data = ScrapedData(
        ML_vars,
        DirVars(["BTC"]),
        data_type="crypto",
    )
    extra_data = ExtraStockData(stock_data)
    dates = Dates(ML_vars, extra_data)
    print(dates.last_date)
    print(dates.future_trading_days)
