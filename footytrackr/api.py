#!/usr/bin/env python3
"""
Footytrackr API - Simple REST API for player value predictions
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Footytrackr API", version="0.1.0")

# Load the trained model
MODEL_PATH = Path("artifacts/ridge_model_v3.joblib")
try:
    model = joblib.load(MODEL_PATH)
    print(f"Loaded model from {MODEL_PATH}")
except FileNotFoundError:
    print(f"Warning: Model not found at {MODEL_PATH}")
    model = None

# Load residual quantiles for prediction intervals
# (computed by scripts/09_prediction_intervals.py)
_PI_PATH = Path("artifacts/prediction_interval_summary_v3.json")
try:
    with open(_PI_PATH) as f:
        _pi = json.load(f)
    _Q10: float = _pi["residual_quantiles_log"]["q10"]
    _Q90: float = _pi["residual_quantiles_log"]["q90"]
    _PI_COVERAGE: float = _pi["empirical_coverage"]
    print(
        f"Loaded prediction intervals from {_PI_PATH} "
        f"(empirical coverage: {_PI_COVERAGE:.1%})"
    )
except FileNotFoundError:
    print(f"Warning: Prediction interval summary not found at {_PI_PATH}")
    _Q10, _Q90, _PI_COVERAGE = -0.0, 0.0, 0.0


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
    interval_coverage: float


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
    df = pd.DataFrame([features.model_dump()])

    # Add derived features (similar to feature engineering)
    for window in ["w180", "w365"]:
        games = df[f"{window}_games_played"]
        minutes = df[f"{window}_minutes_played"]
        goals = df[f"{window}_goals"]
        assists = df[f"{window}_assists"]
        yellow = df[f"{window}_yellow_cards"]
        red = df[f"{window}_red_cards"]

        # Per-90 rates
        nineties = minutes / 90.0
        df[f"{window}_goals_per90"] = goals / nineties.replace(0, np.nan)
        df[f"{window}_assists_per90"] = assists / nineties.replace(0, np.nan)
        df[f"{window}_g_plus_a_per90"] = (goals + assists) / nineties.replace(0, np.nan)
        df[f"{window}_yellow_per90"] = yellow / nineties.replace(0, np.nan)
        df[f"{window}_red_per90"] = red / nineties.replace(0, np.nan)
        df[f"{window}_minutes_per_game"] = minutes / games.replace(0, np.nan)
        df[f"{window}_played_any_minutes"] = (minutes > 0).astype(int)

    # Fill NaN with 0 for prediction
    df = df.fillna(0)

    # Make prediction
    log_pred = model.predict(df)[0]
    value_pred = np.exp(log_pred)

    # Prediction interval: shift log prediction by residual quantiles,
    # then back-transform to EUR. Quantiles were derived from training
    # residuals and validated on held-out future data.
    # See artifacts/prediction_interval_summary_v3.json for empirical coverage.
    ci_lower = float(np.exp(log_pred + _Q10))
    ci_upper = float(np.exp(log_pred + _Q90))

    return PredictionResponse(
        predicted_log_value=log_pred,
        predicted_value_eur=value_pred,
        confidence_interval={"lower": ci_lower, "upper": ci_upper},
        interval_coverage=_PI_COVERAGE,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
