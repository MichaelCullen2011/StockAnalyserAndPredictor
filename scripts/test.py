import pandas as pd
import numpy as np
import os

_root = os.getcwd()
_data = os.path.join(_root, "datasets", "scraped")


df = pd.read_csv(os.path.join(_root, "datasets", "crypto", "historical-crypto-price-data.csv"))
print(df)