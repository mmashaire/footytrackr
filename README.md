# Footytrackr

[![CI](https://github.com/mmashaire/footytrackr/actions/workflows/ci.yml/badge.svg)](https://github.com/mmashaire/footytrackr/actions/workflows/ci.yml)

Footytrackr is a football analytics project I built to answer one practical question:

Can we estimate player market value in a way that is simple, honest about uncertainty, and easy to explain?

I used historical Transfermarkt data and treated this like a small real-world data product, not a one-off notebook. The focus is on clean pipelines, reproducible outputs, and decisions that can be defended.
It also serves as a practical applied AI/ML portfolio piece: the emphasis is on interpretable modeling, honest uncertainty, and production-minded delivery rather than black-box hype.

## Recent quality-focused updates

A few recent improvements pushed this further toward a real portfolio-grade data product:

- **Safer prediction API**: requests are validated for realistic age ranges, non-negative football stats, and non-empty identifiers.
- **Consistent error handling**: invalid API payloads now return a stable JSON error shape instead of ad hoc failures.
- **Model reporting workflow**: `footytrackr model-report` generates a compact reliability summary from saved evaluation artifacts.
- **Test coverage for the new paths**: API, CLI, and reporting logic are all regression-tested.

## Why I built this

Most portfolio projects stop at model accuracy. I wanted to go further and show how I think when working through messy data end to end:

- Building versioned features that can be compared over time
- Avoiding leakage with strict time-based splits
- Measuring model behavior, not just headline scores
- Turning single-value predictions into useful prediction ranges
- Documenting trade-offs clearly

This project reflects how I like to work: practical, transparent, and iterative.

## Model Card

For detailed documentation on model architecture, training data, performance, limitations, and ethics considerations, see **[MODELCARD.md](MODELCARD.md)**.

Key points:

- **Algorithm**: Ridge regression on log-transformed market value
- **Training**: 396k samples (2003–2022); test on future data (2022–2025)
- **Performance**: 34% better than baselines; ~€2.34M MAE; calibrated prediction intervals included
- **Honest uncertainty**: Every prediction includes a 90% confidence interval
- **Limitations documented**: Systematic optimism for high-value players, geographic bias, league-dependent performance

## What the project contains

### 1) Data pipeline

Raw football data is cleaned and transformed into model-ready tables with versioned scripts.

Core feature sets include:

- Rolling windows over 180 and 365 days
- Per-90 rates for goals, assists, and cards
- Age and position context
- Club and league context

Outputs are stored as versioned CSVs so results can be reproduced and compared:

- data/features/player_value_features_v1.csv
- data/features/player_value_features_v2.csv
- data/features/player_value_features_v3.csv

### 2) Market value model

The model is intentionally simple: Ridge regression on log-transformed market value.

Why this choice:

- It gives strong enough performance for this use case
- Coefficients remain interpretable
- It is easy to debug and communicate to non-ML audiences

Each training run produces saved artifacts, including:

- Metrics summaries
- Top coefficients
- Ablation results
- Trained model files

### 3) Diagnostics and error analysis

Beyond MAE and RMSE, the project checks where the model is wrong and why.

Included analyses:

- Error by position
- Error by league
- Over- and under-prediction cases
- Calibration by decile

This helps answer engineering questions such as:

- Is the model consistently biased in certain groups?
- Are strong average metrics hiding weak subgroup performance?

### 4) Bias correction experiments

Because targets are modeled in log space, back-transforming to EUR can introduce bias.

I tested:

- Global smearing correction
- Groupwise smearing correction

The goal here is not to claim a perfect correction, but to make the error profile visible and measurable.

### 5) Prediction intervals

Point estimates can look precise while being misleading.

So the project also outputs prediction intervals (for example, central 80 percent) using residual quantiles from training data and validates their real coverage on future periods.

This is the part I consider most practical for real usage.

## Current results at a glance

- Ridge beats simple baselines by a clear margin
- Context features add meaningful signal on top of pure performance stats
- Bias from log back-transformation is real and quantifiable
- Prediction intervals are well calibrated on held-out future data
- The latest `v3` report shows **~EUR 2.34M MAE** with **80.8% empirical coverage** against an 80% target

The model is not meant to be flashy. It is meant to be reliable, readable, and useful.

## What to try first

If you only spend five minutes with this repo, these are the best entry points:

1. Run a player prediction with explanations:
   ```bash
   footytrackr predict --input-file examples/young_striker_gb1.json --explain --scout-assistant
   ```
2. Generate the monitoring-style model summary:
   ```bash
   footytrackr model-report
   ```
3. Open `artifacts/model_report_v3.md` for a compact reliability snapshot.

## Repository layout

- scripts/: data prep, feature engineering, model training, diagnostics
- data/: raw, processed, and feature data
- artifacts/: metrics, coefficients, ablations, interval summaries
- visuals/: charts used in evaluation and communication
- notebooks/: exploratory analysis
- footytrackr/: API and package code
- tests/: checks for feature-building logic
- examples/: sample player JSON files for testing predictions

## How to run it

Install dependencies:

```bash
pip install -r requirements.txt
pip install -e ".[dev,api]"
```

Run the packaged CLI:

```bash
footytrackr build-features --version v3
footytrackr train
footytrackr model-report
footytrackr test tests/test_api.py
footytrackr api --reload
```

### Try a prediction

Use pre-built example players or build your own JSON. Examples are in `examples/`:

```bash
# Young striker in English Premier League
footytrackr predict --input-file examples/young_striker_gb1.json

# Experienced midfielder in Spanish league
footytrackr predict --input-file examples/experienced_midfielder_es1.json

# Prospect winger in Luxembourg league
footytrackr predict --input-file examples/young_winger_l1.json

# Veteran defender in Italian league
footytrackr predict --input-file examples/veteran_defender_it1.json
```

Or pass JSON inline:

```bash
footytrackr predict --json '{"age": 25, "position": "Centre-Forward", "w180_games_played": 15, "w180_minutes_played": 1200, "w180_goals": 8, "w180_assists": 3, "w180_yellow_cards": 2, "w180_red_cards": 0, "w365_games_played": 30, "w365_minutes_played": 2500, "w365_goals": 14, "w365_assists": 6, "w365_yellow_cards": 4, "w365_red_cards": 0, "player_club_domestic_competition_id": "GB1"}'
```

Include local explainability output (top positive and negative drivers):

```bash
footytrackr predict --input-file examples/young_striker_gb1.json --explain
```

Run Scout Assistant mode for a plain-English scouting summary:

```bash
footytrackr predict --input-file examples/young_striker_gb1.json --scout-assistant
```

### Generate a model report

For a recruiter-friendly summary of model reliability, run:

```bash
footytrackr model-report
```

This writes `artifacts/model_report_v3.json` and `artifacts/model_report_v3.md`, pulling together overall error, interval coverage, directional bias, and the highest-risk subgroups in one place.

At the moment, the report surfaces a few useful realities instead of hiding them:

- overall error is still material (about **EUR 2.34M MAE** on the held-out split)
- interval coverage is close to the target (**80.8% vs 80.0%**)
- the model still tends to **underpredict overall**
- league-level analysis needs richer competition labels than the current `Unknown` heavy artifact

### API validation notes

Prediction requests are intentionally validated before they reach the model. The API rejects:

- ages outside the 16-50 range
- negative minutes, goals, assists, or cards
- empty or whitespace-only categorical fields

That keeps demo usage cleaner and makes the project feel more production-minded than a bare notebook wrapper.

### Scout Assistant demo

If you only read one output in this project, start with this. It combines point estimate, uncertainty, and model reasoning in one response.

```bash
footytrackr predict --input-file examples/young_striker_gb1.json --explain --scout-assistant
```

Example response (shape):

```json
{
  "predicted_log_value": 13.04,
  "predicted_value_eur": 460132.71,
  "confidence_interval": {
    "lower": 112345.89,
    "upper": 987654.32
  },
  "interval_coverage": 0.8079,
  "explanation": {
    "baseline_log_value": 12.5,
    "top_positive": [
      {
        "feature": "w180_goals",
        "feature_value": 8.0,
        "transformed_feature": "w180_goals",
        "contribution_log": 0.9
      }
    ],
    "top_negative": [
      {
        "feature": "age",
        "feature_value": 25.0,
        "transformed_feature": "age",
        "contribution_log": -0.2
      }
    ]
  },
  "scout_assistant": {
    "summary": "Estimated market value is EUR 460K with a 90% interval from EUR 112K to EUR 988K. This sits in the 'Development or depth option' tier with high uncertainty.",
    "valuation_band": "Development or depth option",
    "uncertainty_level": "high",
    "confidence_note": "Treat this as a broad range estimate and rely on additional scouting evidence.",
    "key_positives": ["w180_goals"],
    "key_risks": ["age"]
  }
}
```

Run the API with existing artifacts:

```bash
uvicorn footytrackr.api:app --host 0.0.0.0 --port 8000 --reload
```

Request prediction explanations from the API by adding `explain=true`:

```bash
curl -X POST "http://127.0.0.1:8000/predict?explain=true" \
  -H "Content-Type: application/json" \
  -d @examples/young_striker_gb1.json
```

PowerShell equivalent:

```powershell
$body = Get-Content .\examples\young_striker_gb1.json -Raw
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/predict?explain=true" -ContentType "application/json" -Body $body | ConvertTo-Json -Depth 8
```

Request Scout Assistant output by adding `scout_assistant=true`:

```bash
curl -X POST "http://127.0.0.1:8000/predict?scout_assistant=true" \
  -H "Content-Type: application/json" \
  -d @examples/young_striker_gb1.json
```

PowerShell equivalent:

```powershell
$body = Get-Content .\examples\young_striker_gb1.json -Raw
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/predict?scout_assistant=true" -ContentType "application/json" -Body $body | ConvertTo-Json -Depth 8
```

Rebuild the full pipeline (feature versions + training + analysis):

```bash
python scripts/02_build_value_features.py
python scripts/02_build_value_features_v2.py
python scripts/02_build_value_features_v3.py
python scripts/03_train_market_value.py
python scripts/06_error_analysis.py
python scripts/07_bias_correction.py
python scripts/08_groupwise_bias_correction.py
python scripts/09_prediction_intervals.py
```

Optional: build local DuckDB for faster analysis:

```bash
python scripts/04_build_duckdb.py
```

## Security and public repo notes

- Copy `.env.example` to `.env` and set your own values before running Docker.
- `.env`, local data, model binaries, and DuckDB files are git-ignored.
- No production credentials are stored in this repository.

## Tech stack

- Python
- pandas, numpy, scikit-learn
- duckdb
- matplotlib
- Jupyter

No AutoML, no heavy framework dependency, and no black-box modeling choices in the core pipeline. The AI/ML signal here is applied and explainable, not performative.

## Notes for recruiters and engineers

If you are reviewing this for hiring:

- Recruiters: this project shows ownership across data, modeling, evaluation, and communication.
- Engineers: scripts are versioned, artifacts are explicit, and decisions are traceable.
- AI/ML hiring managers: the project demonstrates applied machine learning with leakage-safe evaluation, calibration, uncertainty estimation, and an API that serves reproducible predictions.

I am happy to walk through design trade-offs, what I would productionize next, and what I would change with more time.

## Contributing

Interested in improving this project? See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on reporting issues, suggesting features, and submitting pull requests.

The project welcomes thoughtful contributions that:

- Strengthen reproducibility or documentation
- Improve data engineering practices
- Add tests or diagnostics
- Fix bugs or security issues

## Disclaimer

Transfermarkt values are market estimates, not objective player quality.
This project models patterns in those estimates and should be interpreted in that context.
