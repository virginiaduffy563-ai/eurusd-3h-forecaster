
from fastapi import FastAPI

app = FastAPI(
    title="EUR/USD 3-Hour Forecast API",
    version="1.0.0"
)

@app.get("/")
def root():
    return {
        "service": "EUR/USD 3-Hour Forecast API",
        "status": "online",
        "forecast_engine": "ready"
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "forecast_engine": "online"
    }
