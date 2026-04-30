# Footytrackr

[![CI](https://github.com/mmashaire/footytrackr/actions/workflows/ci.yml/badge.svg)](https://github.com/mmashaire/footytrackr/actions/workflows/ci.yml)

**A portfolio project demonstrating end-to-end data engineering, applied ML, and production software practices.**

Footytrackr answers one practical question: *Can we estimate player market value in a way that is simple, reproducible, honest about uncertainty, and easy to explain?*

Using historical Transfermarkt data, I built this as a real-world data product—not a notebook. The project emphasizes clean pipelines, reproducible artifacts, interpretable modeling, and decisions that hold up to scrutiny. Every component was designed to be production-ready and defensible in an interview.

## What This Project Demonstrates

### Data & ML  
- **Leakage-safe evaluation**: Strict time-based splits; no information leakage  
- **Interpretable modeling**: Ridge regression on engineered features, not black-box complexity  
- **Honest uncertainty**: Every prediction includes calibrated confidence intervals  
- **Comprehensive diagnostics**: Error analysis by position, league, and decile; bias detection and correction  
- **Reproducible artifacts**: Versioned features, metrics, and model outputs—compare and iterate safely

### Software Engineering  
- **Type safety**: Strict mypy configuration; all public code type-hinted  
- **Automated testing**: 41+ tests covering data pipelines, API endpoints, CLI, and reporting  
- **CI/CD pipeline**: Black, isort, mypy, and pytest on every push  
- **Input validation**: API rejects unrealistic ages, negative stats, and empty fields  
- **Consistent error handling**: Stable JSON error shapes; clear failure messages  
- **Docker support**: Reproducible environment; easy deployment

### Code Quality  
- **Small, focused functions**: Easy to test, debug, and reason about  
- **Clear documentation**: Model Card details architecture, limitations, and ethics considerations  
- **Production-grade API**: FastAPI with Pydantic schemas, request validation, and local explainability  
- **Clean CLI**: Intuitive subcommands for data pipelines, training, reporting, and predictions

## Key Results

- **~€2.34M MAE** on held-out 2022–2025 data  
- **80.8% empirical coverage** against 80% target (well-calibrated intervals)  
- **34% better than naive baselines** (median or mean player value)  
- **Leakage-free**: Trained on 2003–2022; evaluated on future data only  
- **Interpretable**: Top coefficients clearly identify value drivers per position and context

The model is not flashy. It is designed to be reliable, transparent, and useful.

## Model Architecture & Design Decisions

### Why Ridge Regression?

Interpretability matters. Ridge regression keeps coefficients human-readable so stakeholders can understand what drives valuations. The log-transformed target space is justified because market values span orders of magnitude (€100K to €300M+); linear regression on the log scale is stable and explainable.

**Performance trade-off**: Ridge performs ~34% better than naive baselines (using median or mean player value) while remaining simple enough to debug and defend.

### Feature Engineering

The model uses rolling windows (180 and 365 days) to capture form momentum:
- **Per-90 rates**: Goals, assists, cards—normalized for playing time  
- **Context**: Age, position, club, league  
- **Recency**: Shorter windows weight recent performance  

All features are versioned (v1, v2, v3) so you can compare engineering decisions.

### Uncertainty Quantification

Every prediction includes a **90% confidence interval** derived from residual quantiles on training data. On held-out 2022–2025 data:
- **Target coverage**: 80%  
- **Actual coverage**: 80.8%  
- **Coverage gap**: +0.8% (well calibrated)

This is more useful than a point estimate alone.

### Bias & Diagnostics

Because the model operates in log space, back-transformation can introduce bias. The project:
1. **Detects bias** by group (position, league, valuation tier)  
2. **Quantifies bias** with signed error analysis  
3. **Tests corrections** (global and groupwise smearing)  
4. **Reports honestly** about limitations (see [MODELCARD.md](MODELCARD.md))

The goal is not perfection—it's visibility into where the model struggles.

## Project Structure

```
footytrackr/
├── scripts/           # Data pipeline: cleaning, features, training, diagnostics
├── data/
│   ├── raw/          # Original Transfermarkt snapshots
│   ├── processed/    # Cleaned tables
│   └── features/     # Versioned feature sets (v1, v2, v3)
├── footytrackr/      # Package code
│   ├── api.py        # FastAPI prediction server
│   ├── cli.py        # Command-line interface
│   └── reporting.py  # Model diagnostics
├── tests/            # 41+ regression tests
├── artifacts/        # Trained models, metrics, reports
├── examples/         # Sample player JSON files
└── MODELCARD.md      # Full model documentation
```

### Data Pipeline

Raw Transfermarkt data flows through:
1. **Cleaning** (`clean_players.py`, `clean_club_games.py`)  
2. **Feature engineering** (`02_build_value_features_v3.py`)  
3. **Model training** (`03_train_market_value.py`)  
4. **Diagnostics** (`06_error_analysis.py`, `07_bias_correction.py`, `09_prediction_intervals.py`)  
5. **Reporting** (`10_model_report.py`)

Each stage produces versioned outputs so you can audit and reproduce results.

## Quick Start

### Install

```bash
pip install -r requirements.txt
pip install -e ".[dev,api]"
```

### Try a Prediction (CLI)

**Quick prediction:**
```bash
footytrackr predict --input-file examples/young_striker_gb1.json --scout-assistant
```

**With feature explanations:**
```bash
footytrackr predict --input-file examples/young_striker_gb1.json --explain
```

**Inline JSON:**
```bash
footytrackr predict --json '{"age": 25, "position": "Centre-Forward", "w180_games_played": 15, "w180_minutes_played": 1200, "w180_goals": 8, "w180_assists": 3, "w180_yellow_cards": 2, "w180_red_cards": 0, "w365_games_played": 30, "w365_minutes_played": 2500, "w365_goals": 14, "w365_assists": 6, "w365_yellow_cards": 4, "w365_red_cards": 0, "player_club_domestic_competition_id": "GB1"}'
```

### Try the API

```bash
uvicorn footytrackr.api:app --reload
curl -X POST "http://localhost:8000/predict?explain=true" \
  -H "Content-Type: application/json" \
  -d @examples/young_striker_gb1.json
```

### Generate Model Report

```bash
footytrackr model-report
```
Outputs `artifacts/model_report_v3.md` with error analysis, bias assessment, and risk flags.

## Full API Usage & Examples

### Prediction with Scout Assistant (Recommended for First Use)

The Scout Assistant output combines point estimate, uncertainty, and explainability:

```bash
footytrackr predict --input-file examples/young_striker_gb1.json --explain --scout-assistant
```

Example output:
```json
{
  "predicted_value_eur": 460132.71,
  "confidence_interval": {"lower": 112345.89, "upper": 987654.32},
  "interval_coverage": 0.8079,
  "explanation": {
    "top_positive": [{"feature": "w180_goals", "feature_value": 8.0, "contribution_log": 0.9}],
    "top_negative": [{"feature": "age", "feature_value": 25.0, "contribution_log": -0.2}]
  },
  "scout_assistant": {
    "summary": "Estimated market value is EUR 460K with a 90% interval from EUR 112K to EUR 988K.",
    "valuation_band": "Development or depth option",
    "uncertainty_level": "high",
    "key_positives": ["w180_goals"],
    "key_risks": ["age"]
  }
}
```

### REST API

Start the server:
```bash
uvicorn footytrackr.api:app --host 0.0.0.0 --port 8000 --reload
```

**Health check** (validates model & prediction intervals are ready):
```bash
curl http://localhost:8000/health
# Returns {"status": "healthy", "model_loaded": true}
# Or {"status": "unhealthy", "reason": "model_not_loaded"} if artifacts missing
```

Request with explanations:
```bash
curl -X POST "http://localhost:8000/predict?explain=true&scout_assistant=true" \
  -H "Content-Type: application/json" \
  -d @examples/young_striker_gb1.json
```

PowerShell:
```powershell
$body = Get-Content .\examples\young_striker_gb1.json -Raw
Invoke-RestMethod -Method Post -Uri "http://localhost:8000/predict?explain=true&scout_assistant=true" \
  -ContentType "application/json" -Body $body | ConvertTo-Json -Depth 10
```

### Model Reporting

Generate a diagnostic report with error breakdown and risk assessment:
```bash
footytrackr model-report
```

Outputs `artifacts/model_report_v3.md` (human-readable) and `artifacts/model_report_v3.json` (structured).

### Running the Full Pipeline

To retrain from scratch:
```bash
# Feature engineering (v1, v2, v3)
footytrackr build-features --version v3

# Training
footytrackr train

# Analysis & reporting
footytrackr error-analysis
footytrackr bias-correction
footytrackr groupwise-bias-correction
footytrackr prediction-intervals
footytrackr model-report
```

Or run all at once:
```bash
footytrackr pipeline
```

## Documentation

- **[MODELCARD.md](MODELCARD.md)**: Full model documentation (architecture, data, performance, limitations, ethics)
- **[CONTRIBUTING.md](CONTRIBUTING.md)**: Contributing guidelines and code standards
- **[CI Workflow](.github/workflows/ci.yml)**: Type checking, linting, and testing on every push

## Tech Stack

- **Core**: Python 3.9+, pandas, numpy, scikit-learn
- **Data**: duckdb, Transfermarkt snapshots
- **API**: FastAPI, Pydantic, uvicorn
- **Quality**: black, isort, mypy (strict), pytest
- **Deployment**: Docker, GitHub Actions CI
- **Visualization**: matplotlib, seaborn

## Design Philosophy

This project prioritizes **transparent engineering over hype**:
- ✓ Interpretable models over black-box accuracy chasing
- ✓ Versioned artifacts over hidden state
- ✓ Leakage-free evaluation over inflated metrics
- ✓ Calibrated uncertainty over false confidence
- ✓ Clear trade-offs over overselling

## For Hiring Managers & Code Reviewers

**Data engineering signal**: Versioned pipelines, explicit feature engineering, reproducible artifacts, time-based splits with no leakage.

**ML signal**: Leakage-safe evaluation, calibration analysis, bias detection, uncertainty quantification, comparison against baselines.

**Software signal**: Type safety (strict mypy), automated testing (41+ tests), CI/CD, clean API design, error handling, production-ready code.

**Communication signal**: Model Card documenting limitations, clear README explaining decisions, honest assessment of where the model fails.

I'm happy to discuss design trade-offs, what I'd productionize next, or architectural decisions in more detail.

## Disclaimer

Transfermarkt market values are estimates, not ground truth. This model captures patterns in those estimates; interpretation should reflect that.
