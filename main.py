import os
import sys
import time
from pathlib import Path

from flask import Flask, request
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(os.path.join(BASE_DIR, "scripts", "prediction_scripts"))
sys.path.append(os.path.join(BASE_DIR, "scripts", "prediction_scripts", "data_models"))
sys.path.append(os.path.join(BASE_DIR, "gcp"))


import gcp_missing_dates as scrape_to_gcp
from scrapers.stock_scraper import path_check
from get_gcp import connect
from date_model import Dates
from dir_model import DirVars
from extra_stock_model import ExtraStockData
from ml_vars_model import MLVars
from scraped_model import ScrapedData

# problems with importing tensorflow to docker images so avoiding that for now
# from model_create_predict import create_and_predict, model_exists


app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent


@app.route("/")
def hello():
    path = path_check()
    return f"Hello World! - {path}"


@app.route("/stock_scraper", methods=["POST", "GET"])
def scrape_stocks():
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
    client = connect()
    for symbol in symbols:
        scrape_to_gcp.run(client, symbol, data_type="stocks")
    return f"Finished Scraping {symbols}"


@app.route("/crypto_scraper")
def scrape_crypto():
    coins = [
        "BTC",
        "ETH",
        "ADA",
        "NANO",
        "XMR",
        "LRC",
        # "BNB",
        # "SOL",
        # "UNI",
        # "SHIB",
        # "DOGE",
        # "DOT",
        # "ALGO",
    ]
    client = connect()
    for coin in coins:
        scrape_to_gcp.run(client, coin, data_type="crypto")
    return f"Finished Scraping {coins}"


# @app.route("/train_model", methods=["POST", "GET"])
# def train_model():
#     if request.method == "POST":
#         symbols = request.body["symbols"]
#         timescale = request.body["timescale"]
#         data_type = request.body["data_type"]
#         print(f"Training models for {symbols} on a {timescale} timescale")
#     else:
#         data_type = "stocks"
#         symbols = (
#             [
#                 "AAPL",
#                 "TSLA",
#                 "GME",
#                 "ABNB",
#                 "PLTR",
#                 "ETSY",
#                 "ENPH",
#                 "GOOG",
#                 "AMZN",
#                 "IBM",
#                 "DIA",
#                 "IVV",
#                 "NIO",
#             ]
#             if data_type == "stocks"
#             else [
#                 "BTC",
#                 "ETH",
#                 "ADA",
#                 "NANO",
#                 "XMR",
#                 "BNB",
#                 "SOL",
#                 "UNI",
#                 "SHIB",
#                 "DOGE",
#                 "DOT",
#                 "ALGO",
#                 "LRC",
#             ]
#         )
#         timescale = "days"
#     start_time = time.time()

#     ML = MLVars(timescale=timescale, future=1, data_type=data_type)
#     dir_vars = DirVars(symbols=symbols)
#     extra_data = ExtraStockData(
#         stock_data=ScrapedData(ML=ML, dir=dir_vars, data_type=data_type)
#     )
#     dates = Dates(ML=ML, extra_data=extra_data)

#     string_date = pd.to_datetime(dates.last_date).strftime("%Y%m%d")

#     for symbol in symbols:
#         """define training and predict bools based on if a model already exists"""
#         model_name = f"20211208-{symbol}-{ML.timescale}-{ML.epochs}epochs-extracol"
#         train_bool = False if model_exists(model_name) else True
#         predict_future = False if train_bool else True
#         print(
#             f"\n {symbol} \n {string_date} \n Train\Validate: {train_bool} \n Predict: {predict_future} \n"
#         )

#         """ run the training/predicting script """
#         create_and_predict(
#             symbol=symbol,
#             heuristics={
#                 "ML": ML,
#                 "dir_vars": dir_vars,
#                 "extra_data": extra_data,
#                 "dates": dates,
#                 "model_name": model_name,
#                 "train_bool": train_bool,
#                 "predict_future": predict_future,
#             },
#         )

#     return f"Average model train time for {symbols}: {(time.time()-start_time)/len(symbols)}"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
