import os
from pathlib import Path

from flask import Flask

BASE_DIR = Path(__file__).resolve().parent

import src.gcp.gcp_missing_dates as scrape_to_gcp
from src.scrapers.stock_scraper import path_check
from src.gcp.get_gcp import connect

app = Flask(__name__)


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
    ]
    client = connect()
    for coin in coins:
        scrape_to_gcp.run(client, coin, data_type="crypto")
    return f"Finished Scraping {coins}"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
