import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent


class DirVars:
    """Initialises and creates the data and dir directories"""

    symbols: list
    _dirs = list

    def __init__(self, symbols):
        self.symbols: list = symbols
        self._dirs: list = self.make_dirs()

    def __repr__(self):
        return f"{self._dirs}"

    def make_dirs(self) -> list:
        # dict will be layed out according to --> {_dirs[ticker] = [_plots, _pred_data, _valid_data]}
        _dirs = {ticker: [] for ticker in self.symbols}
        for ticker in _dirs:
            _plots = os.path.join(BASE_DIR, "datasets", "plots", ticker)
            _pred_data = os.path.join(BASE_DIR, "datasets", "predicted", ticker)
            _valid_data = os.path.join(BASE_DIR, "datasets", "validation", ticker)
            _scraped_data = os.path.join(BASE_DIR, "datasets", "scraped", ticker)
            try:
                os.mkdir(_plots)
            except FileExistsError:
                pass  # if folders exist it runs pass only
            try:
                os.mkdir(os.path.join(_plots, "Predictions"))
            except FileExistsError:
                pass  # if folders exist it runs pass only
            try:
                os.mkdir(_pred_data)
            except FileExistsError:
                pass  # if folders exist it runs pass only
            try:
                os.mkdir(_valid_data)
            except FileExistsError:
                pass  # if folders exist it runs pass only
            _dirs[ticker] = [_plots, _pred_data, _valid_data, _scraped_data]
        return _dirs


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
    stock_dirs = DirVars(symbols)
    print(stock_dirs)
