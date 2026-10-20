from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.predict import NewsPredictor


predictor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor

    predictor = NewsPredictor()
    yield

    predictor = None


app = FastAPI(
    title="AG News Classification API",
    description="Classify news using Logistic Regression or DistilBERT.",
    version="1.0.0",
    lifespan=lifespan,
)


class PredictionRequest(BaseModel):
    text: str = Field(min_length=1, max_length=10000)
    model: str = Field(default="distilbert")


@app.get("/health")
def health():
    return {
        "status": "healthy" if predictor is not None else "starting",
        "model_loaded": predictor is not None,
    }


@app.post("/predict")
def predict(request: PredictionRequest):
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Models are not loaded yet.",
        )

    text = request.text.strip()

    if not text:
        raise HTTPException(
            status_code=422,
            detail="Text cannot be empty.",
        )

    if request.model == "logistic_regression":
        result = predictor.predict_with_logistic_regression(text)

    elif request.model == "distilbert":
        result = predictor.predict_with_distilbert(text)

    else:
        raise HTTPException(
            status_code=400,
            detail="Model must be 'logistic_regression' or 'distilbert'.",
        )

    return result
