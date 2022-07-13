import os
import time
import sys

sys.path.append(os.path.join(os.getcwd(), "scripts"))
import datetime

import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import pandas_gbq as gbq

import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler

import model_validation as validation_csv
import model_comparison as comparison_csv

from data_models.date_model import Dates
from data_models.dir_model import DirVars
from data_models.ml_vars_model import MLVars
from data_models.scraped_model import ScrapedData
from data_models.extra_stock_model import ExtraStockData


# import initialisations
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# yf.pdr_override()

mpl.rcParams["figure.figsize"] = (8, 6)
mpl.rcParams["axes.grid"] = False

# some base directories
# BASE_DIR = os.path.dirname(__file__)
_root = os.getcwd()
_models = os.path.join(_root, "models")


def quick_shift(df, N=1):
    """Quick implementation of dataframe shift"""
    shift_value = np.roll(df.values, N, axis=0)
    shift_value[0:N] = np.NaN
    return pd.DataFrame(shift_value, index=df.index, columns=df.columns)


pd.DataFrame.quick_shift = quick_shift


def save_predicted_gcp(
    df: pd.DataFrame, symbol: str, data_type: str = "stocks", timescale: str = "days"
) -> None:
    """updates the gcp sql server by appending the passed through df"""
    db = (
        f"crypto_prices_predicted_{timescale}"
        if data_type == "crypto"
        else f"stock_prices_predicted_{timescale}"
    )
    # currently does not consider the extra cols schemas
    gbq.to_gbq(
        df,
        f"{db}.{symbol}",
        "stock-price-database",
        if_exists="replace",
    )
    print("Updated!")


def model_exists(model_name) -> bool:
    """checks if the model has already been trained and saved"""
    return os.path.exists(os.path.join(_models, model_name))


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True) -> pd.DataFrame:
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()

    # input sequence (t-n, ... t-1)
    time_c = time.time()
    for i in range(n_in, 0, -1):
        cols.append(df.quick_shift(i))
        names += [("var%d(t-%d)" % (j + 1, i)) for j in range(n_vars)]

    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.quick_shift(-i))
        if i == 0:
            names += [("var%d(t)" % (j + 1)) for j in range(n_vars)]
        else:
            names += [("var%d(t+%d)" % (j + 1, i)) for j in range(n_vars)]

    # concat together
    agg = pd.concat((col.T for col in cols), axis=0).T
    agg.columns = names

    # # drop rows with NaN values HAS LARGE SLOW DOWN ON PREDICTION LOOPS
    if dropnan:
        agg = agg.dropna()
    return agg


def verify_model_predictions(input_data, input_objs, input_vars):
    x_test, date_time, y_test = input_data
    model, scaler = input_objs
    symbol, batch, num_features, n, extra_data, date_time = input_vars

    yhat = model.predict(x_test)
    x_test = x_test.reshape((x_test.shape[0], batch * num_features))

    x_test_and_pred = np.concatenate((yhat, x_test[:, 1 - num_features :]), axis=1)
    inv_yhat = scaler.inverse_transform(x_test_and_pred)

    y_test = y_test.reshape((len(y_test), 1))
    inv_y = np.concatenate((y_test, x_test[:, 1 - num_features :]), axis=1)
    inv_y = scaler.inverse_transform(inv_y)

    predicted_data = pd.DataFrame({name: [] for name in extra_data.column_names})
    actual_data = pd.DataFrame({name: [] for name in extra_data.column_names})

    valid_plot = pd.DataFrame({"time": [], "close": [], "predicted": []})
    predicted_data["time"], actual_data["time"] = date_time.tolist(), date_time.tolist()

    predicted_data = predicted_data[-inv_yhat.shape[0] :]
    actual_data = actual_data[-inv_yhat.shape[0] :]

    i = 0
    for name in extra_data.column_names:
        predicted_data[name] = inv_yhat[:, i]
        actual_data[name] = inv_y[:, i]
        i += 1

    predicted_data = predicted_data.drop_duplicates(subset=["time"])
    actual_data = actual_data.drop_duplicates(subset=["time"])

    valid_plot["time"], valid_plot["close"], valid_plot["predicted"] = (
        pd.to_datetime(predicted_data["time"]),
        actual_data["close"],
        predicted_data["close"],
    )

    predicted_data.set_index("time", inplace=True)
    actual_data.set_index("time", inplace=True)

    # for the min timescale, reduce the scope of the plot for better granularity and to remove timescale as the plotting index
    valid_plot_plot = valid_plot.copy()
    if ML.timescale == "mins":
        valid_plot_plot["time"] = valid_plot_plot["time"].dt.strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        valid_plot_plot = valid_plot_plot[
            -60 * 16 * 2 :
        ]  # grabs last 2 days of min data
    else:
        valid_plot_plot["time"] = valid_plot_plot["time"].dt.strftime("%Y-%m-%d")

    valid_plot_plot.set_index("time", inplace=True)
    valid_plot.set_index("time", inplace=True)

    # plot
    fig2, ax2 = plt.subplots()
    ax2.set_title(
        f"{symbol} Predicted Prices for {ML.timescale} from historical data: "
    )
    ax2.set_xlabel(f"Dates")
    ax2.set_ylabel(f"Price (USD)")
    splits = (
        int(valid_plot_plot.shape[0] / 100) if valid_plot_plot.shape[0] > 100 else 1
    )
    ax2.plot(valid_plot_plot[["close"]][::splits])
    ax2.plot(valid_plot_plot[["predicted"]][::splits], marker="x")
    ax2.legend(["Actual", "Predicted"], loc="lower right")
    plt.xticks(rotation=90)

    # plt.show()
    return predicted_data, actual_data, valid_plot


def prediction_loop(input_data, input_objs, input_vars):
    x_total, predicted_data, df = input_data
    scaler, model = input_objs
    batch, num_features, extra_data = input_vars

    test_data = x_total[-1:]  # latest data point
    yhat = model.predict(test_data)

    test_data = test_data.reshape((test_data.shape[0], batch * num_features))
    test_data = np.concatenate((yhat, test_data[:, 1 - num_features :]), axis=1)
    # print("shape for inverse transform \n", test_data.shape)

    i = 0
    for name in extra_data.column_names:
        predicted_data[name] = scaler.inverse_transform(test_data)[:, i]
        i += 1

    # predicted_data['close'], predicted_data['open'], predicted_data['high'], predicted_data['low'], predicted_data['volume'] = scaler.inverse_transform(test_data)[:,0], scaler.inverse_transform(test_data)[:,1], scaler.inverse_transform(test_data)[:,2], scaler.inverse_transform(test_data)[:,3], scaler.inverse_transform(test_data)[:,4]

    # print("\n predicted df \n", predicted_data)
    df = pd.concat([df, predicted_data])
    try:
        df = df.set_index(["time"])
    except KeyError:
        pass

    return predicted_data, df


def transform_data(dataframe, n, num_features, ML):
    # transform and scale dataframe values
    values = dataframe.values
    values = values.astype("float32")

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_values = scaler.fit_transform(values)
    total_df = series_to_supervised(data=scaled_values, n_in=ML.batch)

    # split the data into train and test values
    total_values = total_df.values
    # previous definition of n is based on before series_to_supervised transformation
    n = total_values.shape[0]
    train_values = total_values[: int(n * 0.7), :]
    test_values = total_values[int(n * 0.7) :, :]
    print(total_values.shape, train_values.shape, test_values.shape)

    # split the train and test values into x and y (x is the input and y is the expected output)
    x_train, y_train = (
        train_values[:, : ML.batch * num_features],
        train_values[:, -num_features],
    )
    x_train = x_train.reshape((x_train.shape[0], ML.batch, num_features))

    x_test, y_test = (
        test_values[:, : ML.batch * num_features],
        test_values[:, -num_features],
    )
    x_test = x_test.reshape((x_test.shape[0], ML.batch, num_features))

    x_total, y_total = (
        total_values[:, : ML.batch * num_features],
        total_values[:, -num_features],
    )
    x_total = x_total.reshape((x_total.shape[0], ML.batch, num_features))
    return x_train, y_train, x_test, y_test, x_total, y_total, scaler


def create_and_predict(symbol: list, heuristics: dict):
    ML = heuristics["ML"]
    dir_vars = heuristics["dir_vars"]
    extra_data = heuristics["extra_data"]
    dates = heuristics["dates"]
    model_name = heuristics["model_name"]
    train_bool = heuristics["train_bool"]
    predict_future = heuristics["predict_future"]
    data_type = heuristics["data_type"]

    ## setting all the variables
    df = extra_data.extra_cols_dataframe[symbol]
    date_time = extra_data.date_time[symbol]
    valid_file_name = (
        f"{symbol}-{ML.timescale}-{ML.epochs}epochs-extracol-validation.csv"
    )

    validate, train_model = train_bool, train_bool

    n = df.shape[0]
    num_features = df.shape[1]

    total_dates = pd.to_datetime(
        date_time.append(pd.Series(dates.future_trading_days)), utc=True
    ).reset_index(drop=True)

    ## transform our data and get data for training and plotting
    x_train, y_train, x_test, y_test, x_total, y_total, scaler = transform_data(
        dataframe=df, n=n, num_features=num_features, ML=ML
    )

    ## define and train the model
    if train_model:
        model = tf.keras.Sequential(
            [
                tf.keras.layers.LSTM(
                    units=50,
                    return_sequences=False,
                    input_shape=(x_train.shape[1], x_train.shape[2]),
                ),
                tf.keras.layers.Dense(units=1),
            ]
        )
        # train the model
        model.compile(optimizer="adam", loss="mean_squared_error")
        history = model.fit(
            x_train,
            y_train,
            epochs=ML.epochs,
            batch_size=ML.batch,
            validation_data=(x_test, y_test),
            verbose=1,
            shuffle=True,
        )
        # save the model
        model.save(os.path.join(_models, model_name))
        # quick plot of train losses vs test losses
        fig1, ax1 = plt.subplots()
        ax1.plot(history.history["loss"], label="train")
        ax1.plot(history.history["val_loss"], label="test")
        ax1.set_xlabel(f"Number of Epochs")
        ax1.set_ylabel(f"Loss Value")
        ax1.set_title(f"{symbol} train loss vs test loss: ")

    ## verify model performance compared to actual data
    model = tf.keras.models.load_model(os.path.join(_models, model_name))
    if validate:
        print("Verifying Models Predictions...")
        input_data = [x_test, date_time, y_test]
        input_objs = [model, scaler]
        input_vars = [symbol, ML.batch, num_features, n, extra_data, date_time]
        predicted_data, actual_data, valid_plot = verify_model_predictions(
            input_data, input_objs, input_vars
        )

        # add to validation csv (to compare between models)
        valid_plot.to_csv(os.path.join(dir_vars._dirs[symbol][2], valid_file_name))
        validation_csv.run(valid_file_name, symbol)

        ## writes up historical accuracy of model
        comparison_csv.run(valid_file_name, symbol)

    ## make new predictions
    if predict_future:
        print("Predicting New Close Prices...")
        predicted_data = pd.DataFrame({name: [] for name in extra_data.column_names})

        actual_df = df.copy()
        start = time.time()
        for i in range(ML.num_predictions):
            if i != 0:
                total_values = series_to_supervised(
                    data=scaler.fit_transform(df.values[-500:]),
                    n_in=ML.batch,
                ).values

                x_total_test = total_values[:, : ML.batch * num_features]
                x_total = x_total_test.reshape(
                    (x_total_test.shape[0], ML.batch, num_features)
                )[-1:]
            input_data = [x_total, predicted_data, df]
            input_objs = [scaler, model]
            input_vars = [ML.batch, num_features, extra_data]
            prediction_data, df = prediction_loop(input_data, input_objs, input_vars)
            # print(f"prediction for {ML.timescale} {i+1} of {ML.num_predictions}!")
        df = df.reset_index(drop=True)
        print(f"Time to make all predictions: {time.time() - start}s")

        df["time"], actual_df["time"] = total_dates, total_dates[: -ML.num_predictions]
        # df, actual_df = df.set_index(["time"]), actual_df.set_index(["time"])

        fig3, ax3 = plt.subplots()
        ax3.set_title(
            f"{symbol}s Historical Data and {ML.future} Days Worth of New Predicted Prices: "
        )
        ax3.set_xlabel(f"Dates")
        ax3.set_ylabel(f"Price (USD)")
        df, actual_df = (
            df[-ML.num_predictions * 51 :],
            actual_df[-ML.num_predictions * 50 :],
        )
        splits = int(df.shape[0] / 100) if df.shape[0] > 100 else 1
        ax3.plot(df["close"][int(0.5 * df.shape[0]) :: splits], marker="x")
        ax3.plot(actual_df["close"][int(0.5 * df.shape[0]) :: splits])
        ax3.legend(["predicted", "actual"], loc="lower right")
        fig3.savefig(
            os.path.join(
                dir_vars._dirs[symbol][0],
                "Predictions",
                f"{string_date}-{symbol}-{ML.timescale}-{ML.num_predictions}points-extracol",
            )
        )

        df.to_csv(os.path.join(dir_vars._dirs[symbol][1], valid_file_name))

        # saves the predicted dataframe
        save_predicted_gcp(
            df=df, symbol=symbol, data_type=data_type, timescale=ML.timescale
        )

        if ML.timescale == "days":
            df.reset_index(inplace=True)
            try:
                df2 = pd.read_csv(
                    os.path.join(
                        dir_vars._dirs[symbol][1], f"{symbol}-comparison-data.csv"
                    )
                )
            except FileNotFoundError:
                df2 = pd.DataFrame(columns=df.columns.insert(0, "predicted_on"))

            predicted_rows = df[-ML.future :].values.tolist()
            insert_date = datetime.date.today().strftime("%Y-%m-%d")
            for row in predicted_rows:
                row.insert(0, insert_date)
                df2 = df2.append(
                    pd.Series(
                        row,
                        index=df.columns.insert(0, "predicted_on"),
                    ),
                    ignore_index=True,
                )

            df2.set_index("predicted_on", inplace=True)
            df2.to_csv(
                os.path.join(dir_vars._dirs[symbol][1], f"{symbol}-comparison-data.csv")
            )

            df.set_index("time", inplace=True)

        print(
            f"\n prediction data (last {5 + ML.future} values): \n {df.tail(5 + ML.future)}"
        )
        actual_df["time"] = date_time
        actual_df = actual_df.set_index("time")
        print(f"\n actual data (last 5 values): \n {actual_df.tail(5)}")


if __name__ == "__main__":
    start_time = time.time()
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
    symbols = ["BTC"]
    data_type = "crypto" if symbols[0] == "BTC" else "stocks"

    ML = MLVars(timescale="days", future=15, data_type=data_type)
    dir_vars = DirVars(symbols=symbols)
    extra_data = ExtraStockData(
        stock_data=ScrapedData(ML=ML, dir=dir_vars, data_type=data_type)
    )
    dates = Dates(ML=ML, extra_data=extra_data)

    string_date = pd.to_datetime(dates.last_date).strftime("%Y%m%d")
    print("last data point: ", string_date)
    for symbol in symbols:
        """define training and predict bools based on if a model already exists"""
        model_name = f"20211208-{symbol}-{ML.timescale}-{ML.epochs}epochs-extracol"
        train_bool = False if model_exists(model_name) else True
        predict_future = False if train_bool else True
        # predict_future = True
        print(
            f"\n {symbol} \n {string_date} \n Train\Validate: {train_bool} \n Predict: {predict_future} \n"
        )

        """ run the training/predicting script """
        create_and_predict(
            symbol=symbol,
            heuristics={
                "ML": ML,
                "dir_vars": dir_vars,
                "extra_data": extra_data,
                "dates": dates,
                "model_name": model_name,
                "train_bool": train_bool,
                "predict_future": predict_future,
                "data_type": data_type,
            },
        )
        tf.keras.backend.clear_session()

    print(f"Average model run time: {(time.time()-start_time)/len(symbols)}")
    plt.show()
