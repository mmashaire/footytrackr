#!/usr/bin/env python3
"""
Footytrackr API - Simple REST API for player value predictions
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

app = FastAPI(title="Footytrackr API", version="0.1.0")

# Load the trained model
MODEL_PATH = Path("artifacts/ridge_model_v3.joblib")
try:
    model = joblib.load(MODEL_PATH)
    print(f"Loaded model from {MODEL_PATH}")
except FileNotFoundError:
    print(f"Warning: Model not found at {MODEL_PATH}")
    model = None

class PlayerFeatures(BaseModel):
    age: float
    position: str
    w180_games_played: float
    w180_minutes_played: float
    w180_goals: float
    w180_assists: float
    w180_yellow_cards: float
    w180_red_cards: float
    w365_games_played: float
    w365_minutes_played: float
    w365_goals: float
    w365_assists: float
    w365_yellow_cards: float
    w365_red_cards: float
    player_club_domestic_competition_id: str

class PredictionResponse(BaseModel):
    predicted_log_value: float
    predicted_value_eur: float
    confidence_interval: dict

@app.get("/")
def read_root():
    return {"message": "Welcome to Footytrackr API", "version": "0.1.0"}

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictionResponse)
def predict_player_value(features: PlayerFeatures):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Convert to DataFrame
    df = pd.DataFrame([features.dict()])

    # Add derived features (similar to feature engineering)
    for window in ['w180', 'w365']:
        games = df[f'{window}_games_played']
        minutes = df[f'{window}_minutes_played']
        goals = df[f'{window}_goals']
        assists = df[f'{window}_assists']
        yellow = df[f'{window}_yellow_cards']
        red = df[f'{window}_red_cards']

        # Per-90 rates
        nineties = minutes / 90.0
        df[f'{window}_goals_per90'] = goals / nineties.replace(0, np.nan)
        df[f'{window}_assists_per90'] = assists / nineties.replace(0, np.nan)
        df[f'{window}_g_plus_a_per90'] = (goals + assists) / nineties.replace(0, np.nan)
        df[f'{window}_yellow_per90'] = yellow / nineties.replace(0, np.nan)
        df[f'{window}_red_per90'] = red / nineties.replace(0, np.nan)
        df[f'{window}_minutes_per_game'] = minutes / games.replace(0, np.nan)
        df[f'{window}_played_any_minutes'] = (minutes > 0).astype(int)

    # Fill NaN with 0 for prediction
    df = df.fillna(0)

    # Make prediction
    log_pred = model.predict(df)[0]
    value_pred = np.exp(log_pred)

    # Simple confidence interval (placeholder - could be improved)
    ci_lower = value_pred * 0.8
    ci_upper = value_pred * 1.2

    return PredictionResponse(
        predicted_log_value=log_pred,
        predicted_value_eur=value_pred,
        confidence_interval={"lower": ci_lower, "upper": ci_upper}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)