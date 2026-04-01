from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = ROOT / "artifacts"
DEFAULT_MODEL_VERSION = "v3"


def _load_json_artifact(path: Path) -> dict[str, Any]:
    """Load a JSON artifact with a clear error if it is missing."""
    if not path.exists():
        raise FileNotFoundError(
            f"Required artifact not found: {path}. Run the evaluation pipeline first."
        )

    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _load_csv_artifact(path: Path) -> pd.DataFrame:
    """Load a CSV artifact with a clear error if it is missing."""
    if not path.exists():
        raise FileNotFoundError(
            f"Required artifact not found: {path}. Run the evaluation pipeline first."
        )
    return pd.read_csv(path)


def _format_eur(value: float) -> str:
    """Format euro values for a compact Markdown report."""
    magnitude = abs(value)
    if magnitude >= 1_000_000_000:
        return f"EUR {value / 1_000_000_000:.2f}B"
    if magnitude >= 1_000_000:
        return f"EUR {value / 1_000_000:.2f}M"
    if magnitude >= 1_000:
        return f"EUR {value / 1_000:.0f}K"
    return f"EUR {value:.0f}"


def _format_pct(value: float) -> str:
    """Render a proportion as a percentage string."""
    return f"{value * 100:.1f}%"


def _bias_direction(mean_signed_err_eur: float, mae_eur: float) -> str:
    """Classify the overall directional bias in plain English."""
    threshold = max(mae_eur * 0.10, 50_000.0)
    if mean_signed_err_eur <= -threshold:
        return "underprediction"
    if mean_signed_err_eur >= threshold:
        return "overprediction"
    return "roughly_unbiased"


def _prepare_group_rows(
    frame: pd.DataFrame,
    group_column: str,
    *,
    top_n: int = 3,
    exclude_unknown: bool = False,
) -> list[dict[str, Any]]:
    """Normalise grouped error tables into a compact JSON-ready structure."""
    if frame.empty:
        return []

    working = frame.copy()
    if exclude_unknown:
        unknown_mask = working[group_column].astype(str).str.lower().eq("unknown")
        working = working.loc[~unknown_mask]

    if working.empty:
        return []

    median_column = (
        "median_abs_err_eur"
        if "median_abs_err_eur" in working.columns
        else "median_abs_err"
    )
    signed_column = (
        "mean_signed_err_eur"
        if "mean_signed_err_eur" in working.columns
        else "mean_signed_err"
    )

    ordered = working.sort_values(["mae_eur", "n"], ascending=[False, False]).head(top_n)

    rows: list[dict[str, Any]] = []
    for record in ordered.to_dict(orient="records"):
        rows.append(
            {
                "group": str(record[group_column]),
                "samples": int(record["n"]),
                "mae_eur": float(record["mae_eur"]),
                "median_abs_err_eur": float(record[median_column]),
                "mean_signed_err_eur": float(record[signed_column]),
            }
        )
    return rows


def build_model_report(
    model_version: str = DEFAULT_MODEL_VERSION,
    artifacts_dir: Path | None = None,
) -> dict[str, Any]:
    """Build a concise monitoring-style summary from saved evaluation artifacts."""
    artifact_root = artifacts_dir or ARTIFACTS_DIR

    error_summary = _load_json_artifact(
        artifact_root / f"error_analysis_summary_{model_version}.json"
    )
    interval_summary = _load_json_artifact(
        artifact_root / f"prediction_interval_summary_{model_version}.json"
    )
    by_position = _load_csv_artifact(artifact_root / f"error_by_position_{model_version}.csv")
    by_league = _load_csv_artifact(artifact_root / f"error_by_league_{model_version}.csv")

    mae_eur = float(error_summary["mae_eur"])
    mean_signed_err_eur = float(error_summary["mean_signed_err_eur"])
    target_coverage = float(interval_summary["target_coverage"])
    empirical_coverage = float(interval_summary["empirical_coverage"])
    coverage_gap = round(empirical_coverage - target_coverage, 4)
    bias_direction = _bias_direction(mean_signed_err_eur, mae_eur)

    top_position_risks = _prepare_group_rows(by_position, "position_bucket")
    top_league_risks = _prepare_group_rows(
        by_league,
        "domestic_competition_id",
        exclude_unknown=True,
    )

    risk_flags: list[str] = []
    if coverage_gap <= -0.02:
        risk_flags.append(
            "Interval coverage is below target, so the uncertainty band is still a bit too optimistic."
        )
    elif coverage_gap >= 0.02:
        risk_flags.append(
            "Interval coverage is above target, which suggests the current band may be wider than necessary."
        )
    else:
        risk_flags.append(
            "Interval coverage is close to target, which is a good sign for calibration."
        )

    if bias_direction == "underprediction":
        risk_flags.append(
            "Average signed error stays negative, so the model is still underpricing players overall."
        )
    elif bias_direction == "overprediction":
        risk_flags.append(
            "Average signed error stays positive, so the model is still overpricing players overall."
        )
    else:
        risk_flags.append(
            "Average signed error is reasonably balanced at the top level."
        )

    if top_position_risks:
        worst_position = top_position_risks[0]
        risk_flags.append(
            f"Highest position-level error is in {worst_position['group']} profiles, so that subgroup deserves a closer look."
        )

    if top_league_risks:
        worst_league = top_league_risks[0]
        risk_flags.append(
            f"League-level error is highest in {worst_league['group']}, which is worth checking for sample or feature gaps."
        )
    else:
        risk_flags.append(
            "League breakdown is not yet very informative because the current artifact is mostly tagged as 'Unknown'."
        )

    return {
        "model_version": model_version,
        "generated_from": {
            "error_summary": f"error_analysis_summary_{model_version}.json",
            "interval_summary": f"prediction_interval_summary_{model_version}.json",
            "error_by_position": f"error_by_position_{model_version}.csv",
            "error_by_league": f"error_by_league_{model_version}.csv",
        },
        "overall": {
            "n_test_rows": int(error_summary["n_test_rows"]),
            "cutoff_date": error_summary["cutoff_date"],
            "mae_eur": mae_eur,
            "median_abs_err_eur": float(error_summary["median_abs_err_eur"]),
            "mean_signed_err_eur": mean_signed_err_eur,
            "bias_direction": bias_direction,
            "interval": interval_summary["interval"],
            "target_coverage": target_coverage,
            "empirical_coverage": empirical_coverage,
            "coverage_gap": coverage_gap,
        },
        "top_position_risks": top_position_risks,
        "top_league_risks": top_league_risks,
        "risk_flags": risk_flags,
    }


def render_model_report_markdown(report: dict[str, Any]) -> str:
    """Render the consolidated report as compact Markdown for the repo."""
    overall = report["overall"]
    lines = [
        "# Footytrackr model report",
        "",
        f"Model version: `{report['model_version']}`",
        "",
        "## Overall snapshot",
        "",
        f"- Test rows: {overall['n_test_rows']:,}",
        f"- Cutoff date: `{overall['cutoff_date']}`",
        f"- MAE: {_format_eur(overall['mae_eur'])}",
        f"- Median absolute error: {_format_eur(overall['median_abs_err_eur'])}",
        f"- Mean signed error: {_format_eur(overall['mean_signed_err_eur'])} ({overall['bias_direction']})",
        f"- Interval coverage: {_format_pct(overall['empirical_coverage'])} vs target {_format_pct(overall['target_coverage'])}",
        "",
        "## Risk flags",
        "",
    ]

    for flag in report["risk_flags"]:
        lines.append(f"- {flag}")

    lines.extend(["", "## Highest-error positions", ""])
    if report["top_position_risks"]:
        for row in report["top_position_risks"]:
            lines.append(
                f"- **{row['group']}** — MAE {_format_eur(row['mae_eur'])} across {row['samples']} rows"
            )
    else:
        lines.append("- No position breakdown was available.")

    lines.extend(["", "## Highest-error leagues", ""])
    if report["top_league_risks"]:
        for row in report["top_league_risks"]:
            lines.append(
                f"- **{row['group']}** — MAE {_format_eur(row['mae_eur'])} across {row['samples']} rows"
            )
    else:
        lines.append("- League breakdown is currently too sparse to rank meaningfully.")

    return "\n".join(lines) + "\n"


def write_model_report(
    model_version: str = DEFAULT_MODEL_VERSION,
    artifacts_dir: Path | None = None,
) -> tuple[dict[str, Any], Path, Path]:
    """Build the report and save both JSON and Markdown artifacts."""
    artifact_root = artifacts_dir or ARTIFACTS_DIR
    artifact_root.mkdir(parents=True, exist_ok=True)

    report = build_model_report(model_version=model_version, artifacts_dir=artifact_root)
    json_path = artifact_root / f"model_report_{model_version}.json"
    markdown_path = artifact_root / f"model_report_{model_version}.md"

    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    markdown_path.write_text(render_model_report_markdown(report), encoding="utf-8")

    return report, json_path, markdown_path
