# Stock Analysis and Predictor

## Overview

The **Stock Analysis and Predictor** project provides tools for analyzing stock portfolios, backtesting trading strategies, and predicting future stock prices using machine learning models. It integrates data pipelines from Alpha Vantage and Yahoo Finance APIs to fetch stock price data and offers various scripts for analysis and prediction.

## Features

- **Portfolio Analysis**: Gain insights into portfolio diversity and performance.
- **Backtesting Strategies**: Evaluate the success of trading strategies using historical data.
- **Greenline Analysis**: Identify the last safe price of a stock.
- **Resistance and Pivot Points**: Calculate and visualize key stock price levels.
- **Trading View**: Generate a trading view for stocks.
- **Machine Learning Predictions**: Train models to predict future stock prices.

## Prerequisites

- Python 3.x
- Virtual environment setup
- API keys for Alpha Vantage or Yahoo Finance (if using live data)

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/MichaelCullen2011/StockAnalyserAndPredictor.git
   cd StockAnalyserAndPredictor
   ```

2. Set up a virtual environment and install dependencies:
   ```
   python -m venv .venv
   source .venv/bin/activate  # For Mac/Linux
   pip install -r requirements.txt
   ```

## Usage

### 1. Portfolio Analysis

Edit the `data/portfolio_simple.csv` file with your portfolio data and run:
```
python src/analysis/basic_analysis.py
```
This generates an `analysed_portfolio.csv` file with detailed insights.

![Portfolio Analysis](https://github.com/MichaelCullen2011/StockAnalyserAndPredictor/blob/main/src/images/analysed_csv.png?raw=true)

---

### 2. Strategy Backtesting

The backtesting script evaluates the "Red White Blue" strategy. Customize the stock list and start date in the script, then run:
```
python src/analysis/backtest.py
```
![Backtesting Results](https://github.com/MichaelCullen2011/StockAnalyserAndPredictor/blob/main/src/images/backtest.png?raw=true)

---

### 3. Greenline Analysis

Calculate the last solid green line (safe price) for a stock:
```
python src/analysis/greenline.py
```
![Greenline Analysis](https://github.com/MichaelCullen2011/StockAnalyserAndPredictor/blob/main/src/images/greenline.png?raw=true)

---

### 4. Resistance and Pivot Points

Visualize pivot points and resistance levels for a stock. Customize the stock and start date in the script, then run:
```
python src/analysis/resistance_and_pivots.py
```
![Resistance and Pivots](https://github.com/MichaelCullen2011/StockAnalyserAndPredictor/blob/main/src/images/pivots.png?raw=true)

---

### 5. Trading View

Generate a trading view for a stock:
```
python src/analysis/trading_view.py
```
![Trading View](https://github.com/MichaelCullen2011/StockAnalyserAndPredictor/blob/main/src/images/tradingview.png?raw=true)

---

### 6. Predictions

The **Predictions** module allows users to forecast future stock prices using machine learning models. This feature is designed to provide insights into potential price movements based on historical data.

#### How It Works
The prediction system uses a supervised learning approach, based on historical stock price data. It includes the following steps:

1. **Data Collection**: Stock price data is fetched from Alpha Vantage or Yahoo Finance APIs daily. The data includes open, high, low, close prices, and trading volume.
2. **Data Preprocessing**: The raw data is cleaned and transformed into a format suitable for machine learning. This includes:
   - Handling missing values.
   - Normalizing features to ensure consistent scaling.
   - Creating lagged features (e.g., previous days' prices) to capture temporal dependencies.
3. **Feature Engineering**: Additional features such as moving averages, RSI (Relative Strength Index), and Bollinger Bands are computed to provide more features for the model to train on.
4. **Model Training**: The historifcal dataset is then used to train a simple Long Short-Term Memory (LSTM) neural network, which is well-suited for time-series forecasting.
5. **Evaluation**: The model's performance is evaluated using metrics such as Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) on a validation dataset.
6. **Prediction**: Once trained, the model can be used to predict future stock prices. It is however, very underpowered and a simple model. Users will find that it will not predict anything reasonable outside of 2 - 3 days in the future.

#### How to Use
To train the model and generate predictions, run the following command:
```
python src/analysis/predict_stock.py
```

You can customize the script to specify:
- The stock symbol (e.g., AAPL, TSLA).
- The date range for training data.
- Hyperparameters such as the number of LSTM layers, learning rate, and batch size.

#### Example Output
The script generates a plot comparing the actual stock prices with the predicted prices for the test dataset. It also outputs the predicted prices for the next `n` days.

![Prediction Results](https://github.com/MichaelCullen2011/StockAnalyserAndPredictor/blob/main/src/images/predictor_new.png?raw=true)

---

This module provides a wide range of financial asset analysis and data pipeline examples as well as a simple ML model for stock price forecasting.

---

## Contributing

Contributions are welcome! Feel free to fork the repository, make changes, and submit a pull request.

## Contact

**Author**: Michael Cullen  
**Email**: michaelcullen2011@hotmail.co.uk