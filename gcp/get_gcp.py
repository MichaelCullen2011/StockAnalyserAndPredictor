from google.oauth2 import service_account
from google.cloud import bigquery
import pandas as pd
import pandas_gbq as gbq
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
# BASE_DIR = os.path.dirname(__file__)


def connect() -> bigquery.Client:
    """connects to the gcp sql server using credentials.json in root directory"""
    print("Connecting to gcp server...")
    creds = service_account.Credentials.from_service_account_file(
        os.path.join(BASE_DIR, "credentials.json")
    )
    # set gbq credentials
    gbq.context.credentials = creds
    gbq.context.project = creds.project_id
    return bigquery.Client(credentials=creds, project=creds.project_id)


def save_gcp(df: pd.DataFrame, symbol: str, data_type: str = "stocks") -> None:
    """updates the gcp sql server by appending the passed through df"""
    db = "crypto_prices" if data_type == "crypto" else "historical_prices"
    # currently does not consider the extra cols schemas
    gbq.to_gbq(
        df,
        f"{db}.{symbol}",
        "stock-price-database",
        if_exists="append",
    )
    print("Updated!")


def get_from_gcp(client: bigquery.Client, sql_query: str) -> pd.DataFrame:
    """checks if the table exists at gcp sql server and returns the dataframe, if not returns an error"""
    try:
        df = client.query(sql_query)
        df = df.result()
        df = df.to_dataframe()
        return df
    except Exception as e:
        print(e)
        return e


def get_data(
    client: bigquery.Client, symbol: str, data_type: str = "stocks", ts: str = "mins"
) -> pd.DataFrame:
    """chooses the sql query to post to the server"""
    db = "crypto_prices" if data_type == "crypto" else "historical_prices"
    sql_min_query = f"""
		SELECT * FROM `stock-price-database.{db}.{symbol}`
		ORDER BY time DESC
	"""
    sql_day_query = f"""
        SELECT * FROM `stock-price-database.{db}.{symbol}`
        INNER JOIN 
        (
        SELECT 
        MAX(time) max_time, count(time) num_records
        FROM `stock-price-database.{db}.{symbol}`
        GROUP BY Date(`time`)
        ) AS temp
        ON `stock-price-database.{db}.{symbol}`.time = temp.max_time
        ORDER BY time DESC
    """
    date = "2021-11-24"
    sql_date_query = f"""
        SELECT * FROM `stock-price-database.{db}.{symbol}` 
        WHERE time < '{date} 23:59:59' AND time > '{date} 00:00:00'
        ORDER BY time
    """
    sql_query = sql_day_query if ts == "days" else sql_min_query
    df = get_from_gcp(client, sql_query)
    return df


def get_last_gcp_date(
    client: bigquery.Client, symbol: str, data_type: str = "stocks"
) -> pd.DataFrame:
    """returns the last gcp sql date for that symbol"""
    db = "crypto_prices" if data_type == "crypto" else "historical_prices"
    sql_query = f"""
        SELECT * FROM stock-price-database.{db}.{symbol}
        ORDER BY time DESC
        LIMIT 1;
        """
    return get_from_gcp(client, sql_query).index.values[-1]
