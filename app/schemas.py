from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)

    model_config = {"json_schema_extra": {"examples": [{"text": "Film ini sangat bagus dan menarik!"}]}}


class BatchPredictionRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=64)


class SentimentScore(BaseModel):
    label: str
    score: float


class PredictionResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float
    scores: list[SentimentScore]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
