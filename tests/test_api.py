"""
Tests for the FootyTrackr prediction API.

The trained model and prediction interval artefacts are not available in CI,
so we patch them with lightweight stubs so the endpoint logic is tested
independently of the serialised model files.
"""

from pathlib import Path
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
    coefficients[feature_names.index("player_club_domestic_competition_id_GB1")] = 0.3

    ridge = pipeline.named_steps["ridge"]
    ridge.coef_ = coefficients
    ridge.intercept_ = 13.0

    return pipeline


def test_artifact_paths_are_resolved_from_package_root():
    import footytrackr.api as api_module

    repo_root = Path(__file__).resolve().parents[1]
    assert api_module.MODEL_PATH == repo_root / "artifacts" / "ridge_model_v3.joblib"
    assert (
        api_module._PI_PATH
        == repo_root / "artifacts" / "prediction_interval_summary_v3.json"
    )


class TestHealthEndpoint:
    def test_health_returns_200_when_model_healthy(self):
        """Health check succeeds when all artifacts are loaded."""
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
            response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True

    def test_health_returns_unhealthy_when_model_not_loaded(self):
        """Health check returns unhealthy status when model is None."""
        import footytrackr.api as api_module

        with patch.object(api_module, "model", None):
            from footytrackr.api import app

            client = TestClient(app)
            response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "unhealthy"
        assert data["reason"] == "model_not_loaded"

    def test_health_returns_unhealthy_when_prediction_intervals_invalid(self):
        """Health check fails when prediction interval quantiles are not finite."""
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([13.0])

        import footytrackr.api as api_module

        # Test with invalid Q10 (NaN)
        with (
            patch.object(api_module, "model", mock_model),
            patch.object(api_module, "_Q10", np.nan),
            patch.object(api_module, "_Q90", 1.51),
            patch.object(api_module, "_PI_COVERAGE", 0.8079),
        ):
            from footytrackr.api import app

            client = TestClient(app)
            response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "unhealthy"
        assert data["reason"] == "prediction_intervals_not_loaded"

    def test_health_returns_unhealthy_when_prediction_fails(self):
        """Health check fails when model prediction raises an exception."""
        mock_model = MagicMock()
        mock_model.predict.side_effect = RuntimeError("Prediction pipeline broken")

        import footytrackr.api as api_module

        with (
            patch.object(api_module, "model", mock_model),
            patch.object(api_module, "_Q10", -1.43),
            patch.object(api_module, "_Q90", 1.51),
            patch.object(api_module, "_PI_COVERAGE", 0.8079),
        ):
            from footytrackr.api import app

            client = TestClient(app)
            response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "unhealthy"
        assert data["reason"] == "model_prediction_failed"


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


class TestScoutEndpoint:
    """Test the /scout/similar endpoint for finding undervalued players."""

    def _make_scout_features_df(self) -> pd.DataFrame:
        """Create a small scout features dataframe for testing."""
        return pd.DataFrame(
            {
                "player_id": [1, 2, 3, 4, 5],
                "name": [
                    "Messi, Lionel",
                    "Ronaldo, Cristiano",
                    "Neymar Jr.",
                    "Hazard, Eden",
                    "Mbappé, Kylian",
                ],
                "date": pd.to_datetime(
                    [
                        "2025-04-06",
                        "2025-04-06",
                        "2025-04-06",
                        "2025-04-06",
                        "2025-04-06",
                    ]
                ),
                "position": ["Forward", "Forward", "Forward", "Forward", "Forward"],
                "age": [25.0, 24.0, 26.0, 25.0, 25.0],
                "player_club_domestic_competition_id": [
                    "ES1",
                    "IT1",
                    "FR1",
                    "ES1",
                    "FR1",
                ],
                "market_value_in_eur": [
                    100_000_000.0,
                    80_000_000.0,
                    60_000_000.0,
                    50_000_000.0,
                    120_000_000.0,
                ],
                "w365_goals": [8.0, 10.0, 12.0, 4.0, 20.0],
            }
        )

    def test_scout_similar_returns_correct_schema(self):
        """Verify /scout/similar response has required fields."""
        mock_model = MagicMock()
        mock_model.feature_names_in_ = np.array(
            [
                "age",
                "position",
                "player_club_domestic_competition_id",
                "w365_goals",
            ]
        )
        mock_model.predict.side_effect = lambda frame: np.full(len(frame), 13.0)

        import footytrackr.api as api_module

        scout_features = self._make_scout_features_df()

        with (
            patch.object(api_module, "model", mock_model),
            patch.object(api_module, "_scout_features", scout_features),
            patch.object(api_module, "_Q10", -1.43),
            patch.object(api_module, "_Q90", 1.51),
            patch.object(api_module, "_PI_COVERAGE", 0.8079),
        ):
            from footytrackr.api import app

            client = TestClient(app)
            # Query with lowercase name as in lookup
            response = client.get("/scout/similar?player=messi, lionel&n=3")

        assert response.status_code == 200
        body = response.json()

        # Check response structure
        assert "reference_player" in body
        assert "reference_valuation_date" in body
        assert "reference_position" in body
        assert "reference_age" in body
        assert "reference_market_value_eur" in body
        assert "reference_predicted_value_eur" in body
        assert "similar_undervalued" in body
        assert "comparison_criteria" in body

        # Check similar players list
        assert isinstance(body["similar_undervalued"], list)
        assert len(body["similar_undervalued"]) <= 3

        # Check similar player fields
        if body["similar_undervalued"]:
            player = body["similar_undervalued"][0]
            assert "player_id" in player
            assert "name" in player
            assert "valuation_date" in player
            assert "position" in player
            assert "age" in player
            assert "competition" in player
            assert "market_value_eur" in player
            assert "predicted_value_eur" in player
            assert "value_gap_eur" in player
            assert "value_gap_pct" in player

    def test_scout_player_not_found_returns_404(self):
        """Verify 404 when player name not in database."""
        mock_model = MagicMock()
        mock_model.feature_names_in_ = np.array(
            [
                "age",
                "position",
                "player_club_domestic_competition_id",
                "w365_goals",
            ]
        )

        import footytrackr.api as api_module

        scout_features = self._make_scout_features_df()

        with (
            patch.object(api_module, "model", mock_model),
            patch.object(api_module, "_scout_features", scout_features),
        ):
            from footytrackr.api import app

            client = TestClient(app, raise_server_exceptions=False)
            response = client.get("/scout/similar?player=NonexistentPlayer")

        assert response.status_code == 404
        body = response.json()
        assert body["error"]["type"] == "not_found"
        assert "not found" in body["error"]["message"].lower()

    def test_scout_n_parameter_validation(self):
        """Verify n parameter is validated (1-20)."""
        mock_model = MagicMock()

        import footytrackr.api as api_module

        scout_features = self._make_scout_features_df()

        with (
            patch.object(api_module, "model", mock_model),
            patch.object(api_module, "_scout_features", scout_features),
        ):
            from footytrackr.api import app

            client = TestClient(app, raise_server_exceptions=False)

            # Test n=0 (should fail)
            response = client.get("/scout/similar?player=Messi&n=0")
            assert response.status_code == 400

            # Test n=21 (should fail)
            response = client.get("/scout/similar?player=Messi&n=21")
            assert response.status_code == 400

    def test_scout_features_not_loaded_returns_503(self, tmp_path):
        """Verify 503 when scout features are not available."""
        mock_model = MagicMock()
        mock_model.feature_names_in_ = np.array(["age"])

        import footytrackr.api as api_module

        with (
            patch.object(api_module, "model", mock_model),
            patch.object(api_module, "_scout_features", None),
            patch.object(
                api_module, "_FEATURES_PATH", tmp_path / "missing_features.csv"
            ),
            patch.object(api_module, "_PLAYERS_PATH", tmp_path / "missing_players.csv"),
        ):
            from footytrackr.api import app

            client = TestClient(app, raise_server_exceptions=False)
            response = client.get("/scout/similar?player=Messi")

        assert response.status_code == 503
        body = response.json()
        assert body["error"]["type"] == "service_unavailable"

    def test_scout_uses_latest_real_features_for_batch_predictions(self, tmp_path):
        """Predict from each player's latest snapshot in a single model call."""
        import footytrackr.api as api_module

        features_path = tmp_path / "features.csv"
        players_path = tmp_path / "players.csv"
        pd.DataFrame(
            {
                "player_id": [1, 1, 4],
                "date": ["2024-04-01", "2025-04-06", "2025-03-01"],
                "market_value_in_eur": [90_000_000.0, 100_000_000.0, 50_000_000.0],
                "age": [24.0, 25.0, 25.0],
                "position": ["Forward", "Forward", "Forward"],
                "player_club_domestic_competition_id": ["ES1", "ES1", "ES1"],
                "w365_goals": [2.0, 8.0, 4.0],
            }
        ).to_csv(features_path, index=False)
        pd.DataFrame(
            {"player_id": [1, 4], "name": ["Messi, Lionel", "Hazard, Eden"]}
        ).to_csv(players_path, index=False)

        mock_model = MagicMock()
        mock_model.feature_names_in_ = np.array(
            [
                "age",
                "position",
                "player_club_domestic_competition_id",
                "w365_goals",
            ]
        )
        mock_model.predict.side_effect = lambda frame: np.log(
            frame["w365_goals"].to_numpy() + 1
        )

        with (
            patch.object(api_module, "model", mock_model),
            patch.object(api_module, "_scout_features", None),
            patch.object(api_module, "_FEATURES_PATH", features_path),
            patch.object(api_module, "_PLAYERS_PATH", players_path),
        ):
            reference, peers = api_module._find_similar_players("messi, lionel")

        assert mock_model.predict.call_count == 1
        predicted_features = mock_model.predict.call_args.args[0]
        assert predicted_features["w365_goals"].tolist() == [8.0, 4.0]
        assert reference["valuation_date"] == "2025-04-06"
        assert np.isclose(reference["predicted_value_eur"], 9.0)
        assert peers[0]["valuation_date"] == "2025-03-01"
        assert np.isclose(peers[0]["predicted_value_eur"], 5.0)
