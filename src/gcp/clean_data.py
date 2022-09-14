import pandas as pd

pd.options.mode.chained_assignment = None


def clean_df(
    df: pd.DataFrame, extra_col: bool = False, data_type: str = "stocks"
) -> pd.DataFrame:
    cols = (
        ["time", "open", "high", "low", "close", "volume"]
        if not extra_col
        else [
            "time",
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
    )
    cols = (
        [
            "time",
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
        if extra_col and data_type == "crypto"
        else cols
    )
    try:
        df = df[cols]
        df["time"] = pd.to_datetime(df["time"], utc=True)
        df = cols_as_floats(df, cols)
    except KeyError:
        df = df.reset_index()
        df = df[cols]
        df["time"] = pd.to_datetime(df["time"], utc=True)
        df = cols_as_floats(df, cols)
    return df.dropna().drop_duplicates().sort_values("time").set_index("time")


def cols_as_floats(df, cols):
    cols.pop(0)
    df[cols] = df[cols].astype("float64")
    return df


def volume_as_int(df):
    df["volume"] = df["volume"].astype("int64")
    return df


def clean_scraped_crypto_df(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [
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
    ]
    df["time"] = df["time"] * 1000000
    df = df[["time", "open", "high", "low", "close", "volume"]]
    df["time"] = pd.to_datetime(df["time"], infer_datetime_format=True)
    return df
