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
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.metrics import median_absolute_error

tickerData = yf.Ticker("UBS")
df = tickerData.history(period="1d", start="2015-1-1", end="2020-04-20")

# Create some extra features
df["hl_pct"] = (df["High"] - df["Close"]) / df["Close"]
df["delta_pct"] = (df["Close"] - df["Open"]) / df["Close"]

# Moving Average (Weekly, Monthly, Quarterly)
df["5d_uma"] = df["Close"].rolling(5).mean()
df["20d_uma"] = df["Close"].rolling(20).mean()
df["60d_uma"] = df["Close"].rolling(60).mean()

# Exponential Moving Average
df["5d_ema"] = df["Close"].ewm(span=5).mean()
df["20d_ema"] = df["Close"].ewm(span=20).mean()
df["60d_ema"] = df["Close"].ewm(span=60).mean()

# Decide how many days ahead our algorithm should predict
future_forecast = 1  # days into the future
df["label"] = df["Close"].shift(-future_forecast)
df.dropna(inplace=True)

# Training data
X = df.drop(["label"], axis=1)
# X = pd.DataFrame(scale(X))
y = df["label"]


test_size = 20
X_test = X[-test_size:]
y_test = y[-test_size:]
X = X[:-test_size]
y = y[:-test_size]

# Time Series Split:
tscv = TimeSeriesSplit(n_splits=10, max_train_size=None)

# Linear regression model
model = LinearRegression()

# Cross Validate Model
for train_index, val_index in tscv.split(X):
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    print("MAE: ", median_absolute_error(y_val, y_pred))
    print("R² Value: ", model.score(X_val, y_val), "\n")

# Train Model
model.fit(X, y)
y_pred = model.predict(X_test)
print("MAE: ", median_absolute_error(y_test, y_pred))
print("R² Value: ", model.score(X_test, y_test))

# Plot Results
preds = np.empty(len(df))
preds[:] = np.nan
preds[-test_size:] = y_pred
df["preds"] = preds

fig = plt.figure(figsize=(20, 10))
plt.plot(df["Close"], label="reality")
plt.plot(df["preds"], color="r", label="prediction")
plt.legend()
plt.show()

# Show Zoom of prediction
fig = plt.figure(figsize=(20, 10))
plt.plot(df["Close"][-test_size:], label="reality")
plt.plot(df["preds"][-test_size:], color="r", label="prediction")
plt.legend()
plt.show()
