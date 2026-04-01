from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from pydantic import ValidationError


ROOT = Path(__file__).resolve().parents[1]

FEATURE_SCRIPTS = [
    ("v1", "scripts/02_build_value_features.py"),
    ("v2", "scripts/02_build_value_features_v2.py"),
    ("v3", "scripts/02_build_value_features_v3.py"),
]

SCRIPT_COMMANDS = {
    "train": "scripts/03_train_market_value.py",
    "duckdb": "scripts/04_build_duckdb.py",
    "error-analysis": "scripts/06_error_analysis.py",
    "bias-correction": "scripts/07_bias_correction.py",
    "groupwise-bias-correction": "scripts/08_groupwise_bias_correction.py",
    "prediction-intervals": "scripts/09_prediction_intervals.py",
    "model-report": "scripts/10_model_report.py",
}


def _run_python_script(relative_path: str) -> int:
    script_path = ROOT / relative_path
    subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(ROOT),
        check=True,
    )
    return 0


def _run_feature_chain(version: str) -> int:
    for tag, relative_path in FEATURE_SCRIPTS:
        _run_python_script(relative_path)
        if version != "all" and tag == version:
            break
    return 0


def _run_pipeline() -> int:
    _run_feature_chain("all")
    for relative_path in (
        SCRIPT_COMMANDS["train"],
        SCRIPT_COMMANDS["error-analysis"],
        SCRIPT_COMMANDS["bias-correction"],
        SCRIPT_COMMANDS["groupwise-bias-correction"],
        SCRIPT_COMMANDS["prediction-intervals"],
        SCRIPT_COMMANDS["model-report"],
    ):
        _run_python_script(relative_path)
    return 0


def _run_api(host: str, port: int, reload: bool) -> int:
    try:
        import uvicorn
    except ImportError as exc:
        raise ImportError(
            "API dependencies are not installed. Run: pip install -e '.[api]'"
        ) from exc

    uvicorn.run("footytrackr.api:app", host=host, port=port, reload=reload)
    return 0


def _predict_payload(
    payload: dict,
    include_explanation: bool = False,
    include_scout_assistant: bool = False,
) -> dict:
    from footytrackr.api import PlayerFeatures, predict_from_features

    features = PlayerFeatures.model_validate(payload)
    return predict_from_features(
        features,
        include_explanation=include_explanation,
        include_scout_assistant=include_scout_assistant,
    ).model_dump()


def _run_predict(
    raw_json: str | None,
    input_file: str | None,
    explain: bool = False,
    scout_assistant: bool = False,
) -> int:
    if bool(raw_json) == bool(input_file):
        print(
            "Provide exactly one of --json or --input-file for predict.",
            file=sys.stderr,
        )
        return 1

    try:
        if input_file is not None:
            payload = json.loads(Path(input_file).read_text(encoding="utf-8"))
        else:
            payload = json.loads(raw_json)

        prediction = _predict_payload(
            payload,
            include_explanation=explain,
            include_scout_assistant=scout_assistant,
        )
    except FileNotFoundError as exc:
        print(f"Prediction input file not found: {exc.filename}", file=sys.stderr)
        return 1
    except json.JSONDecodeError as exc:
        print(f"Invalid JSON input: {exc.msg}", file=sys.stderr)
        return 1
    except ValidationError as exc:
        print(f"Invalid prediction payload: {exc}", file=sys.stderr)
        return 1
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(json.dumps(prediction, indent=2))
    return 0


def _run_tests(pytest_args: list[str]) -> int:
    try:
        import pytest
    except ImportError as exc:
        raise ImportError(
            "Test dependencies are not installed. Run: pip install -e '.[dev]'"
        ) from exc

    args = pytest_args or ["tests"]
    return int(pytest.main(args))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="footytrackr",
        description="CLI for running Footytrackr data, model, API, and test workflows.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_features = subparsers.add_parser(
        "build-features",
        help="Build feature datasets in dependency order.",
    )
    build_features.add_argument(
        "--version",
        choices=["v1", "v2", "v3", "all"],
        default="all",
        help="Highest feature version to build. Default: all.",
    )

    pipeline = subparsers.add_parser(
        "pipeline",
        help="Run feature building, training, and evaluation scripts.",
    )

    api = subparsers.add_parser("api", help="Start the FastAPI app with Uvicorn.")
    api.add_argument("--host", default="0.0.0.0", help="Host interface to bind.")
    api.add_argument("--port", type=int, default=8000, help="Port to listen on.")
    api.add_argument(
        "--reload",
        action="store_true",
        help="Enable Uvicorn auto-reload for local development.",
    )

    predict = subparsers.add_parser(
        "predict",
        help="Run a local prediction from JSON input without starting the API.",
    )
    predict_group = predict.add_mutually_exclusive_group(required=True)
    predict_group.add_argument(
        "--json",
        dest="raw_json",
        help="Inline JSON payload matching the API request schema.",
    )
    predict_group.add_argument(
        "--input-file",
        help="Path to a JSON file matching the API request schema.",
    )
    predict.add_argument(
        "--explain",
        action="store_true",
        help="Include top positive and negative feature contributions.",
    )
    predict.add_argument(
        "--scout-assistant",
        action="store_true",
        help="Include a scouting-oriented summary with valuation tier and risk level.",
    )

    test = subparsers.add_parser("test", help="Run pytest.")
    test.add_argument(
        "pytest_args",
        nargs=argparse.REMAINDER,
        help="Optional pytest arguments.",
    )

    for command_name in SCRIPT_COMMANDS:
        subparsers.add_parser(command_name, help=f"Run the {command_name} workflow.")

    pipeline.set_defaults(handler=lambda args: _run_pipeline())
    build_features.set_defaults(handler=lambda args: _run_feature_chain(args.version))
    api.set_defaults(handler=lambda args: _run_api(args.host, args.port, args.reload))
    predict.set_defaults(
        handler=lambda args: _run_predict(
            args.raw_json,
            args.input_file,
            explain=args.explain,
            scout_assistant=args.scout_assistant,
        )
    )
    test.set_defaults(handler=lambda args: _run_tests(args.pytest_args))

    for command_name, relative_path in SCRIPT_COMMANDS.items():
        subparser = next(
            action
            for action in subparsers.choices.values()
            if action.prog.endswith(f" {command_name}")
        )
        subparser.set_defaults(
            handler=lambda args, path=relative_path: _run_python_script(path)
        )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        return int(args.handler(args))
    except subprocess.CalledProcessError as exc:
        return exc.returncode or 1
    except ImportError as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())