from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from footytrackr.reporting import build_model_report, render_model_report_markdown


MODEL_VERSION = "v3"


def _write_artifacts(base_dir: Path) -> None:
    (base_dir / f"error_analysis_summary_{MODEL_VERSION}.json").write_text(
        json.dumps(
            {
                "n_test_rows": 120,
                "cutoff_date": "2022-05-18",
                "mae_eur": 2_450_000.0,
                "median_abs_err_eur": 410_000.0,
                "mean_signed_err_eur": -1_250_000.0,
            }
        ),
        encoding="utf-8",
    )

    (base_dir / f"prediction_interval_summary_{MODEL_VERSION}.json").write_text(
        json.dumps(
            {
                "model_version": MODEL_VERSION,
                "interval": "10–90%",
                "target_coverage": 0.80,
                "empirical_coverage": 0.74,
                "n_test": 120,
                "residual_quantiles_log": {
                    "q10": -1.4,
                    "q50": -0.1,
                    "q90": 1.5,
                },
            }
        ),
        encoding="utf-8",
    )

    pd.DataFrame(
        [
            {"position_bucket": "MID", "n": 40, "mae_eur": 3_100_000.0, "median_abs_err": 500_000.0, "mean_signed_err": -2_000_000.0},
            {"position_bucket": "FWD", "n": 35, "mae_eur": 2_800_000.0, "median_abs_err": 450_000.0, "mean_signed_err": -1_500_000.0},
            {"position_bucket": "GK", "n": 20, "mae_eur": 900_000.0, "median_abs_err": 120_000.0, "mean_signed_err": -300_000.0},
        ]
    ).to_csv(base_dir / f"error_by_position_{MODEL_VERSION}.csv", index=False)

    pd.DataFrame(
        [
            {"domestic_competition_id": "Unknown", "n": 120, "mae_eur": 2_450_000.0, "median_abs_err": 410_000.0, "mean_signed_err": -1_250_000.0}
        ]
    ).to_csv(base_dir / f"error_by_league_{MODEL_VERSION}.csv", index=False)


def test_build_model_report_summarises_artifacts_and_flags_risks(tmp_path: Path):
    _write_artifacts(tmp_path)

    report = build_model_report(model_version=MODEL_VERSION, artifacts_dir=tmp_path)

    assert report["model_version"] == MODEL_VERSION
    assert report["overall"]["n_test_rows"] == 120
    assert report["overall"]["bias_direction"] == "underprediction"
    assert report["overall"]["coverage_gap"] == -0.06
    assert report["top_position_risks"][0]["group"] == "MID"
    assert report["top_position_risks"][0]["samples"] == 40
    assert any("coverage" in flag.lower() for flag in report["risk_flags"])
    assert any("league" in flag.lower() for flag in report["risk_flags"])


def test_render_model_report_markdown_includes_key_sections(tmp_path: Path):
    _write_artifacts(tmp_path)
    report = build_model_report(model_version=MODEL_VERSION, artifacts_dir=tmp_path)

    markdown = render_model_report_markdown(report)

    assert "# Footytrackr model report" in markdown
    assert "## Overall snapshot" in markdown
    assert "## Risk flags" in markdown
    assert "MID" in markdown
    assert "underprediction" in markdown.lower()
