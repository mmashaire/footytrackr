#!/usr/bin/env python3
"""
Footytrackr API - Simple REST API for player value predictions
"""

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
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

    @staticmethod
    def _validate_finite_non_negative(value: float, field_label: str) -> float:
        """Reject NaN/inf values and negative football stats."""
        if not np.isfinite(value):
            raise ValueError(f"{field_label} must be a finite number")
        if value < 0:
            raise ValueError(f"{field_label} cannot be negative")
        return value

    @field_validator("age")
    @classmethod
    def validate_age(cls, v: float) -> float:
        """Age must be a realistic value for a professional player."""
        if not np.isfinite(v):
            raise ValueError("Age must be a finite number")
        if not (16 <= v <= 50):
            raise ValueError("Age must be between 16 and 50")
        return v

    @field_validator("position")
    @classmethod
    def validate_position(cls, v: str) -> str:
        """Position must contain a meaningful non-empty label."""
        if not isinstance(v, str):
            raise ValueError("Position must be a non-empty string")
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Position must be a non-empty string")
        return cleaned

    @field_validator(
        "w180_games_played",
        "w365_games_played",
    )
    @classmethod
    def validate_games_played(cls, v: float) -> float:
        """Games played must be finite and non-negative."""
        return cls._validate_finite_non_negative(v, "Games played")

    @field_validator(
        "w180_minutes_played",
        "w365_minutes_played",
    )
    @classmethod
    def validate_minutes_played(cls, v: float) -> float:
        """Minutes played must be finite and non-negative."""
        return cls._validate_finite_non_negative(v, "Minutes played")

    @field_validator(
        "w180_goals",
        "w180_assists",
        "w365_goals",
        "w365_assists",
    )
    @classmethod
    def validate_attacking_stats(cls, v: float) -> float:
        """Goals and assists must be finite and non-negative."""
        return cls._validate_finite_non_negative(v, "Goals and assists")

    @field_validator(
        "w180_yellow_cards",
        "w180_red_cards",
        "w365_yellow_cards",
        "w365_red_cards",
    )
    @classmethod
    def validate_cards(cls, v: float) -> float:
        """Cards must be finite and non-negative."""
        return cls._validate_finite_non_negative(v, "Cards")

    @field_validator("player_club_domestic_competition_id")
    @classmethod
    def validate_competition_id(cls, v: str) -> str:
        """Competition ID must contain a meaningful non-empty label."""
        if not isinstance(v, str):
            raise ValueError("Competition ID must be a non-empty string")
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Competition ID must be a non-empty string")
        return cleaned


class ConfidenceInterval(BaseModel):
    """Prediction interval bounds in EUR."""
    lower: float
    upper: float


class FeatureContribution(BaseModel):
    """A single feature contribution in log-value space."""

    feature: str
    feature_value: str | float | int | None
    transformed_feature: str
    contribution_log: float


class PredictionExplanation(BaseModel):
    """Top positive and negative drivers behind a prediction."""

    baseline_log_value: float
    top_positive: list[FeatureContribution]
    top_negative: list[FeatureContribution]


class ScoutAssistantReport(BaseModel):
    """Decision-support summary for quick scouting interpretation."""

    summary: str
    valuation_band: str
    uncertainty_level: str
    confidence_note: str
    key_positives: list[str]
    key_risks: list[str]


class PredictionResponse(BaseModel):
    """Response schema for player value prediction.

    Includes point prediction, prediction interval with empirical coverage,
    and metadata about the interval estimation.
    """

    predicted_log_value: float
    predicted_value_eur: float
    confidence_interval: ConfidenceInterval
    interval_coverage: float
    explanation: PredictionExplanation | None = None
    scout_assistant: ScoutAssistantReport | None = None


def _http_error_type(status_code: int) -> str:
    """Map HTTP status codes to stable, API-friendly error identifiers."""
    return {
        400: "bad_request",
        404: "not_found",
        422: "validation_error",
        503: "service_unavailable",
    }.get(status_code, "http_error")


@app.exception_handler(RequestValidationError)
async def handle_validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
    """Return request validation failures in a consistent JSON shape."""
    details = []
    for error in exc.errors():
        location = [str(part) for part in error.get("loc", []) if part != "body"]
        details.append(
            {
                "field": ".".join(location) if location else "body",
                "message": error.get("msg", "Invalid value"),
            }
        )

    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "type": "validation_error",
                "message": "Invalid request payload",
                "details": details,
            }
        },
    )


@app.exception_handler(HTTPException)
async def handle_http_error(_: Request, exc: HTTPException) -> JSONResponse:
    """Return predictable error payloads for API clients and CLI callers."""
    message = exc.detail if isinstance(exc.detail, str) else "Request failed"
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "type": _http_error_type(exc.status_code),
                "message": message,
            }
        },
    )


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


def _get_transformed_feature_names() -> list[str]:
    """Return fitted transformed feature names for the loaded pipeline."""
    if model is None:
        raise RuntimeError("Model not loaded")

    preprocessor = model.named_steps["preprocess"]
    names: list[str] = []

    num_features = preprocessor.transformers_[0][2]
    names.extend(list(num_features))

    cat_pipeline = preprocessor.transformers_[1][1]
    onehot = cat_pipeline.named_steps["onehot"]
    cat_features = preprocessor.transformers_[1][2]
    names.extend(list(onehot.get_feature_names_out(cat_features)))

    return names


def _parse_feature_value(
    transformed_feature: str,
    input_frame: pd.DataFrame,
    categorical_features: list[str],
) -> tuple[str, str | float | int | None]:
    """Map a transformed feature back to a user-facing feature/value pair."""
    if transformed_feature in input_frame.columns:
        raw_value = input_frame.iloc[0][transformed_feature]
        if isinstance(raw_value, np.generic):
            raw_value = raw_value.item()
        return transformed_feature, raw_value

    for feature in sorted(categorical_features, key=len, reverse=True):
        prefix = f"{feature}_"
        if transformed_feature.startswith(prefix):
            return feature, transformed_feature[len(prefix):]

    return transformed_feature, None


def _build_prediction_explanation(
    input_frame: pd.DataFrame,
    top_k: int = 3,
) -> PredictionExplanation:
    """Compute local feature contributions for a single prediction."""
    if model is None:
        raise RuntimeError("Model not loaded")

    try:
        preprocessor = model.named_steps["preprocess"]
        ridge = model.named_steps["ridge"]
        transformed_names = _get_transformed_feature_names()
        transformed_row = np.asarray(preprocessor.transform(input_frame)).ravel()
        contributions = transformed_row * ridge.coef_
    except (AttributeError, KeyError, ValueError) as exc:
        raise RuntimeError(
            "Explanation is unavailable for the loaded model artifact"
        ) from exc

    if len(transformed_names) != len(contributions):
        raise RuntimeError("Explanation is unavailable for the loaded model artifact")

    categorical_features = list(preprocessor.transformers_[1][2])
    contribution_frame = pd.DataFrame(
        {
            "transformed_feature": transformed_names,
            "contribution_log": contributions,
        }
    )
    contribution_frame["abs_contribution_log"] = contribution_frame[
        "contribution_log"
    ].abs()

    positive_rows = contribution_frame[contribution_frame["contribution_log"] > 0]
    negative_rows = contribution_frame[contribution_frame["contribution_log"] < 0]

    def _serialize(rows: pd.DataFrame) -> list[FeatureContribution]:
        out: list[FeatureContribution] = []
        for transformed_feature, contribution in rows.nlargest(
            top_k, "abs_contribution_log"
        )[["transformed_feature", "contribution_log"]].itertuples(index=False):
            feature, feature_value = _parse_feature_value(
                transformed_feature,
                input_frame,
                categorical_features,
            )
            out.append(
                FeatureContribution(
                    feature=feature,
                    feature_value=feature_value,
                    transformed_feature=transformed_feature,
                    contribution_log=float(contribution),
                )
            )
        return out

    baseline_log_value = ridge.intercept_
    if isinstance(baseline_log_value, np.generic):
        baseline_log_value = baseline_log_value.item()

    return PredictionExplanation(
        baseline_log_value=float(baseline_log_value),
        top_positive=_serialize(positive_rows),
        top_negative=_serialize(negative_rows),
    )


def _format_value_eur(value: float) -> str:
    """Format a euro value for concise user-facing summaries."""
    if value >= 1_000_000_000:
        return f"EUR {value / 1_000_000_000:.2f}B"
    if value >= 1_000_000:
        return f"EUR {value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"EUR {value / 1_000:.0f}K"
    return f"EUR {value:.0f}"


def _valuation_band(predicted_value_eur: float) -> str:
    """Bucket valuation into scouting-oriented market tiers."""
    if predicted_value_eur < 2_000_000:
        return "Development or depth option"
    if predicted_value_eur < 10_000_000:
        return "Rotation-level asset"
    if predicted_value_eur < 30_000_000:
        return "First-team starter profile"
    if predicted_value_eur < 70_000_000:
        return "High-value starter"
    return "Elite market asset"


def _uncertainty_level(
    predicted_value_eur: float,
    ci_lower: float,
    ci_upper: float,
) -> str:
    """Classify interval width relative to point estimate."""
    if predicted_value_eur <= 0:
        return "high"

    relative_width = (ci_upper - ci_lower) / predicted_value_eur
    if relative_width < 1.5:
        return "low"
    if relative_width < 3.0:
        return "medium"
    return "high"


def _build_scout_assistant_report(
    predicted_value_eur: float,
    ci_lower: float,
    ci_upper: float,
    explanation: PredictionExplanation,
) -> ScoutAssistantReport:
    """Create a plain-English scouting summary from model outputs."""
    uncertainty_level = _uncertainty_level(predicted_value_eur, ci_lower, ci_upper)

    confidence_note_by_level = {
        "low": "Signal is relatively stable for this profile; use as a strong benchmark.",
        "medium": "Signal is directionally useful, but pair with contract and scouting context.",
        "high": "Treat this as a broad range estimate and rely on additional scouting evidence.",
    }

    key_positives = [item.feature for item in explanation.top_positive]
    key_risks = [item.feature for item in explanation.top_negative]

    summary = (
        f"Estimated market value is {_format_value_eur(predicted_value_eur)} "
        f"with a 90% interval from {_format_value_eur(ci_lower)} to "
        f"{_format_value_eur(ci_upper)}. "
        f"This sits in the '{_valuation_band(predicted_value_eur)}' tier "
        f"with {uncertainty_level} uncertainty."
    )

    return ScoutAssistantReport(
        summary=summary,
        valuation_band=_valuation_band(predicted_value_eur),
        uncertainty_level=uncertainty_level,
        confidence_note=confidence_note_by_level[uncertainty_level],
        key_positives=key_positives,
        key_risks=key_risks,
    )


def predict_from_features(
    features: PlayerFeatures,
    include_explanation: bool = False,
    include_scout_assistant: bool = False,
) -> PredictionResponse:
    """Run a validated feature payload through the loaded model."""
    if model is None:
        raise RuntimeError("Model not loaded")

    df = _build_prediction_frame(features)

    log_pred = float(model.predict(df)[0])
    if not np.isfinite(log_pred):
        raise RuntimeError("Model returned an invalid prediction")

    value_pred = float(np.exp(log_pred))
    ci_lower = float(np.exp(log_pred + _Q10))
    ci_upper = float(np.exp(log_pred + _Q90))

    if not all(np.isfinite(value) for value in (value_pred, ci_lower, ci_upper)):
        raise RuntimeError("Prediction interval could not be computed")

    ci_lower, ci_upper = sorted((ci_lower, ci_upper))

    explanation = None
    if include_explanation:
        explanation = _build_prediction_explanation(df)

    scout_assistant = None
    if include_scout_assistant:
        if explanation is None:
            explanation = _build_prediction_explanation(df)
        scout_assistant = _build_scout_assistant_report(
            predicted_value_eur=float(value_pred),
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            explanation=explanation,
        )

    return PredictionResponse(
        predicted_log_value=log_pred,
        predicted_value_eur=value_pred,
        confidence_interval=ConfidenceInterval(lower=ci_lower, upper=ci_upper),
        interval_coverage=_PI_COVERAGE,
        explanation=explanation,
        scout_assistant=scout_assistant,
    )


@app.get("/")
def read_root():
    return {"message": "Welcome to Footytrackr API", "version": "0.1.0"}


@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict_player_value(
    features: PlayerFeatures,
    explain: bool = False,
    scout_assistant: bool = False,
):
    try:
        return predict_from_features(
            features,
            include_explanation=explain,
            include_scout_assistant=scout_assistant,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
