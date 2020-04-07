"""
The code below is used to extract data from a certain stock and apply regression algorithms on these values.

It is based on: https://www.youtube.com/watch?v=JcI5Vnw0b2c&list=PLQVvvaa0QuDfKTOs3Keq_kaG2P55YRn5v&index=2

Author:
fanconic
tria124
"""

import numpy as np
import yfinance as yf
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, scale
from sklearn.model_selection import train_test_split as tts
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression

tickerData = yf.Ticker("UBS")
df = tickerData.history(period='1d', start="2015-1-1", end="2020-04-06")


# Create some extra features
df['hl_pct'] = (df['High'] - df['Close']) / df['Close']
df['delta_pct'] = (df['Close'] - df['Open']) / df['Close']

# Decide how many days ahead our algorithm should predict
future_forecast = 10 # days into the future
df["label"] = df["Close"].shift(-future_forecast)
df.dropna(inplace=True)

# Training data
X = df.drop(["label"], axis = 1)
X = scale(X)
y = df["label"]

# Train-Test Split
X_train, X_test, y_train, y_test = tts(X, y, test_size = 0.2, random_state= 42)


# Regressor
model = RandomForestRegressor(n_estimators=200, random_state=42)
# Train model
model.fit(X_train, y_train)

# Asses model
score  = model.score(X_test, y_test)
print("R² score of Regressor: ", score)

# Plot results
df["Close"].plot()