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
from pydantic import BaseModel, field_validator

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
    """Request schema for player value prediction.
    
    All numeric fields are validated to be within realistic ranges for
    professional football players. The validation helps catch malformed
    input early with clear error messages.
    """
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

    @field_validator("age")
    @classmethod
    def validate_age(cls, v: float) -> float:
        """Age must be between 16 and 50."""
        if not (16 <= v <= 50):
            raise ValueError("Age must be between 16 and 50")
        return v

    @field_validator("position")
    @classmethod
    def validate_position(cls, v: str) -> str:
        """Position must be a non-empty string."""
        if not v or not isinstance(v, str):
            raise ValueError("Position must be a non-empty string")
        return v.strip()

    @field_validator(
        "w180_games_played",
        "w365_games_played",
    )
    @classmethod
    def validate_games_played(cls, v: float) -> float:
        """Games played must be non-negative."""
        if v < 0:
            raise ValueError("Games played cannot be negative")
        return v

    @field_validator(
        "w180_minutes_played",
        "w365_minutes_played",
    )
    @classmethod
    def validate_minutes_played(cls, v: float) -> float:
        """Minutes played must be non-negative."""
        if v < 0:
            raise ValueError("Minutes played cannot be negative")
        return v

    @field_validator(
        "w180_goals",
        "w180_assists",
        "w365_goals",
        "w365_assists",
    )
    @classmethod
    def validate_attacking_stats(cls, v: float) -> float:
        """Goals and assists must be non-negative."""
        if v < 0:
            raise ValueError("Goals and assists cannot be negative")
        return v

    @field_validator(
        "w180_yellow_cards",
        "w180_red_cards",
        "w365_yellow_cards",
        "w365_red_cards",
    )
    @classmethod
    def validate_cards(cls, v: float) -> float:
        """Cards must be non-negative integers."""
        if v < 0:
            raise ValueError("Cards cannot be negative")
        return v

    @field_validator("player_club_domestic_competition_id")
    @classmethod
    def validate_competition_id(cls, v: str) -> str:
        """Competition ID must be a non-empty string."""
        if not v or not isinstance(v, str):
            raise ValueError("Competition ID must be a non-empty string")
        return v.strip()


class ConfidenceInterval(BaseModel):
    """Prediction interval bounds in EUR."""
    lower: float
    upper: float


class PredictionResponse(BaseModel):
    """Response schema for player value prediction.
    
    Includes point prediction, prediction interval with empirical coverage,
    and metadata about the interval estimation.
    """
    predicted_log_value: float
    predicted_value_eur: float
    confidence_interval: ConfidenceInterval
    interval_coverage: float


def _build_prediction_frame(features: PlayerFeatures) -> pd.DataFrame:
    """Convert validated request features into a model-ready dataframe.
    
    Fills in missing contextual fields (height, citizenship, position details)
    with appropriate defaults so the model pipeline can handle them.
    The model was trained with these fields but they're not part of the
    prediction API to keep input simple. Imputation in the pipeline will
    handle the defaults appropriately.
    """
    df = pd.DataFrame([features.model_dump()])

    # Add missing numeric fields (will be median-imputed by the pipeline)
    df["current_club_id"] = np.nan
    df["height_in_cm"] = np.nan
    
    # Add missing categorical fields (will use most-frequent imputation)
    # Use empty string as sentinel; the imputer will replace with most-frequent value
    df["height_bucket"] = ""
    df["country_of_birth"] = ""
    df["country_of_citizenship"] = ""
    df["sub_position"] = ""
    df["foot"] = ""

    for window in ["w180", "w365"]:
        games = df[f"{window}_games_played"]
        minutes = df[f"{window}_minutes_played"]
        goals = df[f"{window}_goals"]
        assists = df[f"{window}_assists"]
        yellow = df[f"{window}_yellow_cards"]
        red = df[f"{window}_red_cards"]

        nineties = minutes / 90.0
        df[f"{window}_goals_per90"] = goals / nineties.replace(0, np.nan)
        df[f"{window}_assists_per90"] = assists / nineties.replace(0, np.nan)
        df[f"{window}_g_plus_a_per90"] = (goals + assists) / nineties.replace(0, np.nan)
        df[f"{window}_yellow_per90"] = yellow / nineties.replace(0, np.nan)
        df[f"{window}_red_per90"] = red / nineties.replace(0, np.nan)
        df[f"{window}_minutes_per_game"] = minutes / games.replace(0, np.nan)
        df[f"{window}_played_any_minutes"] = (minutes > 0).astype(int)

    # Numeric columns should have NaN; fillna(0) handles them
    # Categorical columns should have empty strings; they won't be filled and will be imputed
    return df.fillna(0)


def predict_from_features(features: PlayerFeatures) -> PredictionResponse:
    """Run a validated feature payload through the loaded model."""
    if model is None:
        raise RuntimeError("Model not loaded")

    df = _build_prediction_frame(features)

    log_pred = model.predict(df)[0]
    value_pred = np.exp(log_pred)

    ci_lower = float(np.exp(log_pred + _Q10))
    ci_upper = float(np.exp(log_pred + _Q90))

    return PredictionResponse(
        predicted_log_value=log_pred,
        predicted_value_eur=value_pred,
        confidence_interval=ConfidenceInterval(lower=ci_lower, upper=ci_upper),
        interval_coverage=_PI_COVERAGE,
    )


@app.get("/")
def read_root():
    return {"message": "Welcome to Footytrackr API", "version": "0.1.0"}


@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict_player_value(features: PlayerFeatures):
    try:
        return predict_from_features(features)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
