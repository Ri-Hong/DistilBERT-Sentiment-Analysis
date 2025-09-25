"""
FastAPI service for sentiment analysis using DistilBERT.
"""

from typing import Dict, List

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load the model and tokenizer
MODEL_PATH = "model/artifacts/best"
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

# Create FastAPI app
app = FastAPI(
    title="Sentiment Analysis API",
    description="A service for sentiment analysis using DistilBERT",
    version="1.0.0"
)

class TextInput(BaseModel):
    text: str

class TextsInput(BaseModel):
    texts: List[str]

class PredictionResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float
    model_version: str

class SinglePrediction(BaseModel):
    text: str
    sentiment: str
    confidence: float

class BatchPredictionResponse(BaseModel):
    predictions: List[SinglePrediction]
    model_version: str

class MetricsResponse(BaseModel):
    model_version: str
    model_name: str
    tokenizer_name: str
    max_sequence_length: int

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: TextInput) -> Dict:
    """
    Predict sentiment for input text.
    """
    # Tokenize input
    inputs = tokenizer(
        request.text,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    
    # Get prediction
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=1)
        prediction = torch.argmax(probabilities, dim=1)
        confidence = probabilities[0][prediction.item()].item()
    
    # Map prediction to label
    label = "positive" if prediction.item() == 1 else "negative"
    
    return {
        "text": request.text,
        "sentiment": label,
        "confidence": confidence,
        "model_version": model.config._name_or_path
    }

@app.post("/predict_batch", response_model=BatchPredictionResponse)
async def predict_batch(request: TextsInput) -> Dict:
    """
    Predict sentiment for multiple texts in one request.
    """
    # Tokenize all inputs
    inputs = tokenizer(
        request.texts,
        truncation=True,
        max_length=512,
        padding=True,
        return_tensors="pt"
    )
    
    # Get predictions
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=1)
        predictions = torch.argmax(probabilities, dim=1)
        confidences = [probabilities[i][pred].item() for i, pred in enumerate(predictions)]
    
    # Map predictions to labels
    results = []
    for text, pred, conf in zip(request.texts, predictions, confidences):
        label = "positive" if pred.item() == 1 else "negative"
        results.append(SinglePrediction(
            text=text,
            sentiment=label,
            confidence=conf
        ))
    
    return {
        "predictions": results,
        "model_version": model.config._name_or_path
    }

@app.get("/metrics", response_model=MetricsResponse)
async def metrics() -> Dict:
    """
    Get model performance metrics.
    """
    return {
        "model_version": model.config._name_or_path,
        "model_name": "sentiment-analysis-distilbert",
        "tokenizer_name": tokenizer.name_or_path,
        "max_sequence_length": tokenizer.model_max_length,
    }

@app.get("/health")
async def health() -> Dict:
    """
    Health check endpoint.
    """
    return {"status": "healthy"}
