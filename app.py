import requests
import pandas as pd
pd.options.display.max_columns=200
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import time
from sklearn.dummy import DummyClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("data/btc_data.csv", index_col=0)
data.drop(columns=["market_cap", "difficulty"], inplace=True)
print(data)
