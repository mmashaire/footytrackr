from __future__ import annotations

import json
import subprocess
import sys
import types

import footytrackr.cli as cli


def test_build_features_v2_runs_dependency_chain(monkeypatch):
    calls: list[list[str]] = []

    def fake_run(command, cwd, check):
        calls.append(command)
        assert cwd == str(cli.ROOT)
        assert check is True

    monkeypatch.setattr(cli.subprocess, "run", fake_run)

    exit_code = cli.main(["build-features", "--version", "v2"])

    assert exit_code == 0
    assert calls == [
        [sys.executable, str(cli.ROOT / "scripts/02_build_value_features.py")],
        [sys.executable, str(cli.ROOT / "scripts/02_build_value_features_v2.py")],
    ]


def test_pipeline_runs_expected_scripts_in_order(monkeypatch):
    calls: list[list[str]] = []

    def fake_run(command, cwd, check):
        calls.append(command)

    monkeypatch.setattr(cli.subprocess, "run", fake_run)

    exit_code = cli.main(["pipeline"])

    assert exit_code == 0
    assert calls == [
        [sys.executable, str(cli.ROOT / "scripts/02_build_value_features.py")],
        [sys.executable, str(cli.ROOT / "scripts/02_build_value_features_v2.py")],
        [sys.executable, str(cli.ROOT / "scripts/02_build_value_features_v3.py")],
        [sys.executable, str(cli.ROOT / "scripts/03_train_market_value.py")],
        [sys.executable, str(cli.ROOT / "scripts/06_error_analysis.py")],
        [sys.executable, str(cli.ROOT / "scripts/07_bias_correction.py")],
        [sys.executable, str(cli.ROOT / "scripts/08_groupwise_bias_correction.py")],
        [sys.executable, str(cli.ROOT / "scripts/09_prediction_intervals.py")],
    ]


def test_api_command_invokes_uvicorn(monkeypatch):
    calls: list[tuple[str, str, int, bool]] = []
    fake_uvicorn = types.SimpleNamespace(
        run=lambda app, host, port, reload: calls.append((app, host, port, reload))
    )
    monkeypatch.setitem(sys.modules, "uvicorn", fake_uvicorn)

    exit_code = cli.main(["api", "--host", "127.0.0.1", "--port", "9000", "--reload"])

    assert exit_code == 0
    assert calls == [("footytrackr.api:app", "127.0.0.1", 9000, True)]


def test_predict_command_prints_json(monkeypatch, capsys):
    monkeypatch.setattr(
        cli,
        "_predict_payload",
        lambda payload: {
            "predicted_log_value": 13.0,
            "predicted_value_eur": 442413.0,
            "confidence_interval": {"lower": 105000.0, "upper": 930000.0},
            "interval_coverage": 0.8079,
        },
    )

    exit_code = cli.main(
        [
            "predict",
            "--json",
            json.dumps(
                {
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
            ),
        ]
    )

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["predicted_log_value"] == 13.0
    assert output["confidence_interval"]["lower"] == 105000.0


def test_predict_command_reports_invalid_json(capsys):
    exit_code = cli.main(["predict", "--json", "not-json"])

    assert exit_code == 1
    assert "Invalid JSON input" in capsys.readouterr().err


def test_predict_command_reports_model_errors(monkeypatch, capsys):
    def fake_predict(payload):
        raise RuntimeError("Model not loaded")

    monkeypatch.setattr(cli, "_predict_payload", fake_predict)

    exit_code = cli.main(["predict", "--json", "{}"])

    assert exit_code == 1
    assert "Model not loaded" in capsys.readouterr().err


def test_test_command_invokes_pytest(monkeypatch):
    calls: list[list[str]] = []
    fake_pytest = types.SimpleNamespace(main=lambda args: calls.append(args) or 0)
    monkeypatch.setitem(sys.modules, "pytest", fake_pytest)

    exit_code = cli.main(["test", "tests/test_api.py", "-q"])

    assert exit_code == 0
    assert calls == [["tests/test_api.py", "-q"]]


def test_script_failure_returns_non_zero(monkeypatch):
    def fake_run(command, cwd, check):
        raise subprocess.CalledProcessError(returncode=3, cmd=command)

    monkeypatch.setattr(cli.subprocess, "run", fake_run)

    exit_code = cli.main(["train"])

    assert exit_code == 3