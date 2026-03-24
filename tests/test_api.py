"""
Tests for the FootyTrackr prediction API.

The trained model and prediction interval artefacts are not available in CI,
so we patch them with lightweight stubs so the endpoint logic is tested
independently of the serialised model files.
"""

from unittest.mock import MagicMock, patch

import numpy as np
from fastapi.testclient import TestClient

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

    def test_rejects_empty_competition_id(self):
        client = self._make_client_for_validation_test()
        payload = SAMPLE_PAYLOAD.copy()
        payload["player_club_domestic_competition_id"] = ""
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

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

        assert response.status_code == 503
