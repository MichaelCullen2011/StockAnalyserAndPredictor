import numpy as np
import pandas as pd

from src.gcp.get_gcp import connect, get_data
from src.gcp.clean_data import clean_df


def rsi_calc(df: pd.DataFrame, n: int = 14) -> list:
    deltas = np.diff(df["close"])
    seed = deltas[: n + 1]
    up = seed[seed >= 0].sum() / n
    down = -seed[seed < 0].sum() / n
    rs = up / down
    rsi = np.zeros_like(df["close"])
    rsi[:n] = 100.0 - 100.0 / (1.0 + rs)

    for i in range(n, len(df["close"])):
        delta = deltas[i - 1]  # cause the diff is 1 shorter

        if delta > 0:
            upval = delta
            downval = 0.0
        else:
            upval = 0.0
            downval = -delta

        up = (up * (n - 1) + upval) / n
        down = (down * (n - 1) + downval) / n

        rs = up / down
        rsi[i] = 100.0 - 100.0 / (1.0 + rs)

    return rsi


def extra_cols_calculations(dataframe: pd.DataFrame) -> pd.DataFrame:
    """calculates additional cols for the dataframes (emas, smas, macd, rsi)"""
    # emas
    emas_used = [3, 5, 10, 12, 26, 30]
    smas_used = [10, 30, 50]
    for x in emas_used:
        ema = x
        dataframe["ema_" + str(ema)] = (
            dataframe.iloc[:, 0].ewm(span=ema, adjust=False).mean()
        )
    # smas
    for x in smas_used:
        sma = x
        dataframe["sma_" + str(sma)] = (
            dataframe["close"].ewm(span=sma, adjust=False).mean()
        )
    # macd
    dataframe["macd"] = dataframe["ema_12"] - dataframe["ema_26"]
    # rsi
    dataframe["rsi"] = rsi_calc(dataframe)
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
            "sma_10",
            "sma_30",
            "sma_50",
            "macd",
            "rsi",
        ]
    ]
    return dataframe


def crypto_extra_cols_calculations(dataframe: pd.DataFrame) -> pd.DataFrame:
    """calculates additional cols for the dataframes (emas, smas, macd, rsi)"""
    # emas
    emas_used = [12, 26]
    smas_used = [50, 200]
    for x in emas_used:
        ema = x
        dataframe["ema_" + str(ema)] = (
            dataframe.iloc[:, 0].ewm(span=ema, adjust=False).mean()
        )
    # smas
    for x in smas_used:
        sma = x
        dataframe["sma_" + str(sma)] = (
            dataframe["close"].ewm(span=sma, adjust=False).mean()
        )
    # macd
    dataframe["macd"] = dataframe["ema_12"] - dataframe["ema_26"]
    # rsi
    dataframe["rsi"] = rsi_calc(dataframe, n=8)
    # social media sentiment

    ## add the new dataframe to extra_cols_dataframe
    dataframe = dataframe[
        [
            "close",
            "open",
            "high",
            "low",
            "volume",
            "ema_12",
            "ema_26",
            "sma_50",
            "sma_200",
            "macd",
            "rsi",
        ]
    ]
    return dataframe


def run(symbol: str, data_type: str = "stocks") -> None:
    print(f"Creating extracols dataframe for {symbol}...")
    df = clean_df(get_data(client, symbol, data_type=data_type, ts="days"))
    # error with no numeric types to aggregate?
    extra_cols_df = (
        crypto_extra_cols_calculations(df)
        if data_type == "crypto"
        else extra_cols_calculations(df)
    ).reset_index()
    extra_cols_df = clean_df(extra_cols_df, extra_col=True, data_type=data_type)
    print(extra_cols_df)


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
    symbols = ["AAPL"]
    coins = ["BTC"]
    client = connect()
    for symbol in symbols:
        run(symbol, data_type="stocks")
