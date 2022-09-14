from pathlib import Path
import os
import sys

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(os.path.join(BASE_DIR, "scripts", "prediction_scripts"))
sys.path.append(os.path.join(BASE_DIR, "gcp"))

from data_models.dir_model import DirVars
from data_models.ml_vars_model import MLVars

from get_gcp import connect, get_data
from clean_data import clean_df

import yfinance as yf

yf.pdr_override()


class ScrapedData:
    """scrapes the day data from yahoo finance or min data from av"""

    symbols: list
    _dirs = dict
    timescale = str
    future: int
    data_dict: dict

    def __init__(self, ML, dir, data_type: str = "stocks"):
        self.symbols: list = dir.symbols
        self._dirs: dict = dir._dirs
        self.timescale: str = ML.timescale
        self.future: int = ML.future
        self.data_type = data_type

        self.client = connect()
        self.data_dict: dict = self.grab_data()

    def grab_data(self) -> dict:
        data_dict = {ticker: [] for ticker in self.symbols}
        for ticker in data_dict:
            data_dict[ticker] = clean_df(
                get_data(
                    client=self.client,
                    symbol=ticker,
                    data_type=self.data_type,
                    ts=self.timescale,
                )
            )
        return data_dict


if __name__ == "__main__":
    stock_data = ScrapedData(
        MLVars(timescale="days", data_type="crypto"), DirVars(["AAPL", "GME"])
    )
    print(stock_data.data_dict["AAPL"])
    stock_data_crypto = ScrapedData(
        ML=MLVars(timescale="days"), dir=DirVars(["BTC"]), data_type="crypto"
    )
    print(stock_data_crypto.data_dict["BTC"])
