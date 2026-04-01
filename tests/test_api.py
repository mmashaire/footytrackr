"""
Tests for the FootyTrackr prediction API.

The trained model and prediction interval artefacts are not available in CI,
so we patch them with lightweight stubs so the endpoint logic is tested
independently of the serialised model files.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SAMPLE_PAYLOAD = {
    "age": 25.0,
    "position": "Centre-Forward",
    "w180_games_played": 15.0,
    "w180_minutes_played": 1200.0,
    "w180_goals": 8.0,
    "w180_assists": 3.0,
    "w180_yellow_cards": 2.0,
    "w180_red_cards": 0.0,
    "w365_games_played": 30.0,
    "w365_minutes_played": 2500.0,
    "w365_goals": 14.0,
    "w365_assists": 6.0,
    "w365_yellow_cards": 4.0,
    "w365_red_cards": 0.0,
    "player_club_domestic_competition_id": "GB1",
}


def _make_client(log_pred: float = 13.0, q10: float = -1.43, q90: float = 1.51):
    """Return a TestClient with model and PI artefacts stubbed out."""
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([log_pred])

    import footytrackr.api as api_module

    with (
        patch.object(api_module, "model", mock_model),
        patch.object(api_module, "_Q10", q10),
        patch.object(api_module, "_Q90", q90),
        patch.object(api_module, "_PI_COVERAGE", 0.8079),
    ):
        from footytrackr.api import app

        client = TestClient(app)
        yield client, mock_model


def _make_explainable_model():
    """Create a small fitted pipeline with deterministic feature contributions."""
    import footytrackr.api as api_module

    alt_payload = SAMPLE_PAYLOAD.copy()
    alt_payload.update(
        {
            "age": 34.0,
            "position": "Goalkeeper",
            "w180_goals": 0.0,
            "w180_assists": 0.0,
            "w180_minutes_played": 900.0,
            "w365_goals": 1.0,
            "w365_assists": 0.0,
            "player_club_domestic_competition_id": "IT1",
        }
    )

    row_one = api_module._build_prediction_frame(
        api_module.PlayerFeatures.model_validate(SAMPLE_PAYLOAD)
    )
    row_two = api_module._build_prediction_frame(
        api_module.PlayerFeatures.model_validate(alt_payload)
    )
    X = pd.concat([row_one, row_two], ignore_index=True)
    y = np.array([13.0, 11.5])

    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    pipeline = Pipeline(
        steps=[
            (
                "preprocess",
                ColumnTransformer(
                    transformers=[
                        (
                            "num",
                            Pipeline(
                                steps=[
                                    ("imputer", SimpleImputer(strategy="median")),
                                    ("scaler", StandardScaler()),
                                ]
                            ),
                            numeric_features,
                        ),
                        (
                            "cat",
                            Pipeline(
                                steps=[
                                    (
                                        "imputer",
                                        SimpleImputer(strategy="most_frequent"),
                                    ),
                                    (
                                        "onehot",
                                        OneHotEncoder(
                                            handle_unknown="ignore",
                                            sparse_output=False,
                                        ),
                                    ),
                                ]
                            ),
                            categorical_features,
                        ),
                    ],
                    remainder="drop",
                ),
            ),
            ("ridge", Ridge(alpha=1.0)),
        ]
    )
    pipeline.fit(X, y)

    preprocessor = pipeline.named_steps["preprocess"]
    feature_names = list(preprocessor.transformers_[0][2])
    feature_names.extend(
        preprocessor.transformers_[1][1]
        .named_steps["onehot"]
        .get_feature_names_out(preprocessor.transformers_[1][2])
    )

    coefficients = np.zeros(len(feature_names))
    coefficients[feature_names.index("w180_goals")] = 0.9
    coefficients[feature_names.index("age")] = 0.7
    coefficients[feature_names.index("position_Centre-Forward")] = 0.4
    coefficients[
        feature_names.index("player_club_domestic_competition_id_GB1")
    ] = 0.3

    ridge = pipeline.named_steps["ridge"]
    ridge.coef_ = coefficients
    ridge.intercept_ = 13.0

    return pipeline


class TestHealthEndpoint:
    def test_health_returns_200(self):
        from footytrackr.api import app

        client = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200
        assert "model_loaded" in response.json()


class TestPredictEndpoint:
    def test_predict_returns_200_with_valid_payload(self):
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([13.0])

        import footytrackr.api as api_module

        with (
            patch.object(api_module, "model", mock_model),
            patch.object(api_module, "_Q10", -1.43),
            patch.object(api_module, "_Q90", 1.51),
            patch.object(api_module, "_PI_COVERAGE", 0.8079),
        ):
            from footytrackr.api import app

            client = TestClient(app)
            response = client.post("/predict", json=SAMPLE_PAYLOAD)

        assert response.status_code == 200

    def test_predict_response_schema(self):
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([13.0])

        import footytrackr.api as api_module

        with (
            patch.object(api_module, "model", mock_model),
            patch.object(api_module, "_Q10", -1.43),
            patch.object(api_module, "_Q90", 1.51),
            patch.object(api_module, "_PI_COVERAGE", 0.8079),
        ):
            from footytrackr.api import app

            client = TestClient(app)
            body = client.post("/predict", json=SAMPLE_PAYLOAD).json()

        assert "predicted_log_value" in body
        assert "predicted_value_eur" in body
        assert "confidence_interval" in body
        assert "lower" in body["confidence_interval"]
        assert "upper" in body["confidence_interval"]
        assert "interval_coverage" in body

    def test_predict_ci_uses_residual_quantiles(self):
        """CI bounds must come from residual quantiles, not a ±20% heuristic."""
        log_pred = 13.0
        q10, q90 = -1.43, 1.51
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([log_pred])

        import footytrackr.api as api_module

        with (
            patch.object(api_module, "model", mock_model),
            patch.object(api_module, "_Q10", q10),
            patch.object(api_module, "_Q90", q90),
            patch.object(api_module, "_PI_COVERAGE", 0.8079),
        ):
            from footytrackr.api import app

            client = TestClient(app)
            body = client.post("/predict", json=SAMPLE_PAYLOAD).json()

        expected_lower = float(np.exp(log_pred + q10))
        expected_upper = float(np.exp(log_pred + q90))
        assert abs(body["confidence_interval"]["lower"] - expected_lower) < 1.0
        assert abs(body["confidence_interval"]["upper"] - expected_upper) < 1.0

    def test_predict_ci_lower_less_than_upper(self):
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([13.0])

        import footytrackr.api as api_module

        with (
            patch.object(api_module, "model", mock_model),
            patch.object(api_module, "_Q10", -1.43),
            patch.object(api_module, "_Q90", 1.51),
            patch.object(api_module, "_PI_COVERAGE", 0.8079),
        ):
            from footytrackr.api import app

            client = TestClient(app)
            body = client.post("/predict", json=SAMPLE_PAYLOAD).json()

        assert (
            body["confidence_interval"]["lower"] < body["confidence_interval"]["upper"]
        )

    def test_predict_explanation_is_optional(self):
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([13.0])

        import footytrackr.api as api_module

        with (
            patch.object(api_module, "model", mock_model),
            patch.object(api_module, "_Q10", -1.43),
            patch.object(api_module, "_Q90", 1.51),
            patch.object(api_module, "_PI_COVERAGE", 0.8079),
        ):
            from footytrackr.api import app

            client = TestClient(app)
            body = client.post("/predict", json=SAMPLE_PAYLOAD).json()

        assert body["explanation"] is None
        assert body["scout_assistant"] is None

    def test_predict_explanation_returns_feature_drivers(self):
        explainable_model = _make_explainable_model()

        import footytrackr.api as api_module

        with (
            patch.object(api_module, "model", explainable_model),
            patch.object(api_module, "_Q10", -1.43),
            patch.object(api_module, "_Q90", 1.51),
            patch.object(api_module, "_PI_COVERAGE", 0.8079),
        ):
            from footytrackr.api import app

            client = TestClient(app)
            body = client.post("/predict?explain=true", json=SAMPLE_PAYLOAD).json()

        explanation = body["explanation"]
        assert explanation is not None
        assert explanation["baseline_log_value"] == 13.0

        positive_features = {item["feature"] for item in explanation["top_positive"]}
        negative_features = {item["feature"] for item in explanation["top_negative"]}

        assert "w180_goals" in positive_features
        assert "position" in positive_features
        assert "age" in negative_features

    def test_predict_scout_assistant_returns_summary(self):
        explainable_model = _make_explainable_model()

        import footytrackr.api as api_module

        with (
            patch.object(api_module, "model", explainable_model),
            patch.object(api_module, "_Q10", -1.43),
            patch.object(api_module, "_Q90", 1.51),
            patch.object(api_module, "_PI_COVERAGE", 0.8079),
        ):
            from footytrackr.api import app

            client = TestClient(app)
            body = client.post(
                "/predict?scout_assistant=true",
                json=SAMPLE_PAYLOAD,
            ).json()

        report = body["scout_assistant"]
        assert report is not None
        assert "Estimated market value is" in report["summary"]
        assert report["valuation_band"] in {
            "Development or depth option",
            "Rotation-level asset",
            "First-team starter profile",
            "High-value starter",
            "Elite market asset",
        }
        assert report["uncertainty_level"] in {"low", "medium", "high"}
        assert "w180_goals" in report["key_positives"]


class TestInputValidation:
    """Test that PlayerFeatures validators reject invalid inputs with clear errors."""

    def _make_client_for_validation_test(self):
        """Create a test client for validation tests."""
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([13.0])

        import footytrackr.api as api_module

        with (
            patch.object(api_module, "model", mock_model),
            patch.object(api_module, "_Q10", -1.43),
            patch.object(api_module, "_Q90", 1.51),
            patch.object(api_module, "_PI_COVERAGE", 0.8079),
        ):
            from footytrackr.api import app
            return TestClient(app)

    def test_rejects_negative_age(self):
        client = self._make_client_for_validation_test()
        payload = SAMPLE_PAYLOAD.copy()
        payload["age"] = -5.0
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_rejects_age_too_high(self):
        client = self._make_client_for_validation_test()
        payload = SAMPLE_PAYLOAD.copy()
        payload["age"] = 75.0
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_rejects_negative_games_played(self):
        client = self._make_client_for_validation_test()
        payload = SAMPLE_PAYLOAD.copy()
        payload["w180_games_played"] = -1.0
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_rejects_negative_minutes_played(self):
        client = self._make_client_for_validation_test()
        payload = SAMPLE_PAYLOAD.copy()
        payload["w365_minutes_played"] = -100.0
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_rejects_negative_goals(self):
        client = self._make_client_for_validation_test()
        payload = SAMPLE_PAYLOAD.copy()
        payload["w180_goals"] = -3.0
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_rejects_negative_assists(self):
        client = self._make_client_for_validation_test()
        payload = SAMPLE_PAYLOAD.copy()
        payload["w365_assists"] = -2.0
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_rejects_negative_cards(self):
        client = self._make_client_for_validation_test()
        payload = SAMPLE_PAYLOAD.copy()
        payload["w180_yellow_cards"] = -1.0
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_rejects_empty_position(self):
        client = self._make_client_for_validation_test()
        payload = SAMPLE_PAYLOAD.copy()
        payload["position"] = ""
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_rejects_whitespace_only_position(self):
        client = self._make_client_for_validation_test()
        payload = SAMPLE_PAYLOAD.copy()
        payload["position"] = "   "
        response = client.post("/predict", json=payload)
        body = response.json()

        assert response.status_code == 422
        assert body["error"]["type"] == "validation_error"

    def test_rejects_empty_competition_id(self):
        client = self._make_client_for_validation_test()
        payload = SAMPLE_PAYLOAD.copy()
        payload["player_club_domestic_competition_id"] = ""
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_rejects_whitespace_only_competition_id(self):
        client = self._make_client_for_validation_test()
        payload = SAMPLE_PAYLOAD.copy()
        payload["player_club_domestic_competition_id"] = "   "
        response = client.post("/predict", json=payload)
        body = response.json()

        assert response.status_code == 422
        assert body["error"]["type"] == "validation_error"

    def test_accepts_zero_values(self):
        """Zero is valid for most stats (player may have no minutes/goals yet)."""
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([10.0])

        import footytrackr.api as api_module

        with (
            patch.object(api_module, "model", mock_model),
            patch.object(api_module, "_Q10", -1.43),
            patch.object(api_module, "_Q90", 1.51),
            patch.object(api_module, "_PI_COVERAGE", 0.8079),
        ):
            from footytrackr.api import app
            client = TestClient(app)
            payload = SAMPLE_PAYLOAD.copy()
            payload["w180_games_played"] = 0.0
            payload["w180_goals"] = 0.0
            payload["w365_minutes_played"] = 0.0
            response = client.post("/predict", json=payload)
            assert response.status_code == 200

    def test_accepts_valid_age_boundaries(self):
        """Test age boundaries (16 and 50 inclusive)."""
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([10.0])

        import footytrackr.api as api_module

        with (
            patch.object(api_module, "model", mock_model),
            patch.object(api_module, "_Q10", -1.43),
            patch.object(api_module, "_Q90", 1.51),
            patch.object(api_module, "_PI_COVERAGE", 0.8079),
        ):
            from footytrackr.api import app
            client = TestClient(app)
            
            # Test age 16
            payload = SAMPLE_PAYLOAD.copy()
            payload["age"] = 16.0
            response = client.post("/predict", json=payload)
            assert response.status_code == 200
            
            # Test age 50
            payload = SAMPLE_PAYLOAD.copy()
            payload["age"] = 50.0
            response = client.post("/predict", json=payload)
            assert response.status_code == 200

    def test_predict_503_when_model_not_loaded(self):
        import footytrackr.api as api_module

        with patch.object(api_module, "model", None):
            from footytrackr.api import app

            client = TestClient(app, raise_server_exceptions=False)
            response = client.post("/predict", json=SAMPLE_PAYLOAD)

        body = response.json()

        assert response.status_code == 503
        assert body == {
            "error": {
                "type": "service_unavailable",
                "message": "Model not loaded",
            }
        }

    def test_validation_errors_use_consistent_error_shape(self):
        client = self._make_client_for_validation_test()
        payload = SAMPLE_PAYLOAD.copy()
        payload["w180_goals"] = -1.0

        response = client.post("/predict", json=payload)
        body = response.json()

        assert response.status_code == 422
        assert body["error"]["type"] == "validation_error"
        assert body["error"]["message"] == "Invalid request payload"
        assert body["error"]["details"]
