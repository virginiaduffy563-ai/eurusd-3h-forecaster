from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def root():
    return {
        "status": "online",
        "message": "EUR/USD Forecast API server is running"
    }

@app.get("/health")
def health():
    return {
        "status": "healthy"
    }
