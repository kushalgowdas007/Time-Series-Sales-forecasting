from fastapi import FastAPI

app = FastAPI(
    title="AI Retail Decision Intelligence Platform"
)

@app.get("/")
def home():

    return {
        "message":
        "AI Retail Decision Intelligence Platform API Running"
    }


@app.get("/forecast")
def forecast():

    return {
        "forecast": 32,
        "revenue": 3840,
        "profit": 960,
        "risk": "LOW"
    }

from fastapi import FastAPI
from src.agents.orchestrator import run_pipeline

app = FastAPI()

@app.get("/forecast")
def forecast():

    return run_pipeline()