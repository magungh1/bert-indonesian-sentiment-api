from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.auth import verify_api_key
from app.model import SentimentModel
from app.schemas import (
    BatchPredictionRequest,
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
    SentimentScore,
)

model = SentimentModel()


@asynccontextmanager
async def lifespan(app: FastAPI):
    model.load()
    yield


app = FastAPI(
    title="Indonesian Sentiment Analysis API",
    description="Quantized BERT model for Indonesian text sentiment analysis",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(status="healthy", model_loaded=model.session is not None)


@app.post("/predict", response_model=PredictionResponse, dependencies=[Depends(verify_api_key)])
async def predict_sentiment(request: PredictionRequest):
    result = model.predict(request.text)
    scores = [SentimentScore(label=k, score=v) for k, v in result["scores"].items()]
    return PredictionResponse(
        text=request.text,
        sentiment=result["sentiment"],
        confidence=result["confidence"],
        scores=scores,
    )


@app.post("/predict/batch", response_model=list[PredictionResponse], dependencies=[Depends(verify_api_key)])
async def predict_batch(request: BatchPredictionRequest):
    results = model.predict_batch(request.texts)
    responses = []
    for text, result in zip(request.texts, results):
        scores = [SentimentScore(label=k, score=v) for k, v in result["scores"].items()]
        responses.append(
            PredictionResponse(
                text=text,
                sentiment=result["sentiment"],
                confidence=result["confidence"],
                scores=scores,
            )
        )
    return responses
