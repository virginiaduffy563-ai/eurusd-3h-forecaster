from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import requests
import os

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error

app = FastAPI(
    title="EUR/USD 3-Hour Machine Learning Forecast API",
    version="2.0.0"
)

TWELVE_DATA_API_KEY = os.getenv("TWELVE_DATA_API_KEY")


class ForecastRequest(BaseModel):
    symbol: str = "EUR/USD"
    interval: str = "1h"
    outputsize: int = 500


def get_market_data(symbol: str, interval: str, outputsize: int):
    if not TWELVE_DATA_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="TWELVE_DATA_API_KEY is not configured."
        )

    url = "https://api.twelvedata.com/time_series"

    params = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": outputsize,
        "apikey": TWELVE_DATA_API_KEY,
        "format": "JSON"
    }

    response = requests.get(url, params=params, timeout=20)

    if response.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail="Twelve Data request failed."
        )

    data = response.json()

    if "values" not in data:
        raise HTTPException(
            status_code=502,
            detail=data.get("message", "No market data returned.")
        )

    df = pd.DataFrame(data["values"])

    for column in ["open", "high", "low", "close"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.sort_values("datetime").reset_index(drop=True)

    return df


def create_features(df):
    data = df.copy()

    data["return_1"] = data["close"].pct_change(1)
    data["return_3"] = data["close"].pct_change(3)
    data["return_6"] = data["close"].pct_change(6)
    data["return_12"] = data["close"].pct_change(12)

    data["ema_5"] = data["close"].ewm(span=5).mean()
    data["ema_10"] = data["close"].ewm(span=10).mean()
    data["ema_20"] = data["close"].ewm(span=20).mean()
    data["ema_50"] = data["close"].ewm(span=50).mean()

    delta = data["close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    data["rsi_14"] = 100 - (100 / (1 + rs))

    data["range"] = data["high"] - data["low"]
    data["atr_14"] = data["range"].rolling(14).mean()

    data["volatility_12"] = data["return_1"].rolling(12).std()
    data["volatility_24"] = data["return_1"].rolling(24).std()

    # Three-hour future return target.
    data["future_return_3h"] = (
        data["close"].shift(-3) / data["close"] - 1
    )

    return data


FEATURES = [
    "return_1",
    "return_3",
    "return_6",
    "return_12",
    "ema_5",
    "ema_10",
    "ema_20",
    "ema_50",
    "rsi_14",
    "range",
    "atr_14",
    "volatility_12",
    "volatility_24"
]


def train_model(data):
    training = data.dropna(
        subset=FEATURES + ["future_return_3h"]
    ).copy()

    if len(training) < 150:
        raise HTTPException(
            status_code=422,
            detail="Not enough clean historical data to train the model."
        )

    X = training[FEATURES]
    y = training["future_return_3h"]

    # Keep chronological order. No random shuffle.
    split = int(len(training) * 0.8)

    X_train = X.iloc[:split]
    y_train = y.iloc[:split]

    X_test = X.iloc[split:]
    y_test = y.iloc[split:]

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)

    return model, mae, len(training)


@app.get("/")
def root():
    return {
        "service": "EUR/USD 3-Hour Machine Learning Forecast API",
        "status": "online",
        "forecast_engine": "ready",
        "model": "RandomForestRegressor",
        "forecast_horizon": "3 hours"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "forecast_engine": "online"
    }


@app.post("/predict")
def predict(request: ForecastRequest):

    df = get_market_data(
        request.symbol,
        request.interval,
        request.outputsize
    )

    data = create_features(df)

    model, validation_mae, training_rows = train_model(data)

    latest = data.dropna(subset=FEATURES).iloc[-1]

    X_latest = latest[FEATURES].to_frame().T

    predicted_return = float(model.predict(X_latest)[0])

    current_price = float(latest["close"])

    predicted_price = current_price * (1 + predicted_return)

    if predicted_return > 0.0002:
        direction = "UP"
    elif predicted_return < -0.0002:
        direction = "DOWN"
    else:
        direction = "SIDE"

    return {
        "symbol": request.symbol,
        "forecast_horizon": "3 hours",
        "current_price": current_price,
        "predicted_price_3h": predicted_price,
        "predicted_return": predicted_return,
        "direction": direction,
        "validation_mae": float(validation_mae),
        "training_rows": training_rows,
        "model": "RandomForestRegressor",
        "data_source": "Twelve Data",
        "forecast_status": "MODEL_GENERATED"
    }
