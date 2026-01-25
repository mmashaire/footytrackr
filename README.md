# Footytrackr

**Footytrackr** is a football data project built around historical Transfermarkt data. The goal is to practice realistic data engineering and applied data science workflows using a large, messy, real-world dataset.

## Project Focus

The project focuses on:
- **Building reproducible feature pipelines**
- **Training and evaluating simple, interpretable models**
- **Understanding what actually drives player market value** through analysis, ablation studies, and diagnostics.

> **Note:** This is not a tutorial project. It is intentionally exploratory and iterative.

## Dataset

The data originates from Transfermarkt and includes:
- Player profiles and market values
- Match appearances and minutes played
- Goals, assists, cards, and playing time
- Club, competition, and league context
- Transfer history

The raw dataset contains hundreds of thousands of rows spanning multiple decades and leagues.

## What is implemented
### Feature pipeline

Raw CSVs are transformed into model-ready feature tables using versioned scripts:
- Rolling performance aggregates over 180 and 365 days
- Per-90 performance rates (goals, assists, cards)
- Age, position, physical attributes (where available)
- Club and league context features
- Categorical grouping to control feature cardinality

Feature versions are saved explicitly (v1, v2, v3) to allow comparisons.

**Output:**
```
data/features/player_value_features_v*.csv
```

### Market value model

A linear Ridge regression model is trained to predict log-transformed market value.

**Key design choices:**
- Strict time-based train/test split to prevent leakage
- No direct identifiers or raw euro values used as features
- Global median and position+age baselines for comparison
- Model interpretability preserved through linear coefficients

**Artifacts saved per run:**
- Evaluation metrics (artifacts/metrics_v*.json)
- Coefficient tables (artifacts/ridge_top_coefficients_v*.csv)
- Ablation results (artifacts/ablation_v*.csv)

Trained models (.joblib) are kept local and reproducible, not required for Git.

### Ablation study

An ablation study is used to understand feature reliance.

The same model is trained under different feature restrictions:
- Full feature set
- Without nationality features
- Without nationality and league context
- Performance-only features

This allows direct measurement of how much each feature group contributes to predictive performance.

### Model diagnostics (error analysis & calibration)

Beyond headline metrics, the model is evaluated using:
- Error slicing by position and league
- Top over- and under-predictions
- Residual distribution analysis
- Calibration-by-deciles (mean predicted vs mean actual)

**Artifacts and visuals:**
- artifacts/error_by_position_v*.csv
- artifacts/error_by_league_v*.csv
- artifacts/top_overpredictions_v*.csv
- artifacts/top_underpredictions_v*.csv
- visuals/residual_hist_v*.png
- visuals/pred_vs_actual_log_v*.png
- visuals/calibration_deciles_v*.png

### Log-target bias correction (smearing)

Market values are modeled in log space (log1p(EUR)) to handle heavy-tailed distributions.
Back-transforming predictions to euros introduces bias.

The project includes Duan smearing correction experiments:
- Global smearing factor
- Groupwise smearing by domestic competition (league)

These experiments demonstrate bias–variance trade-offs rather than optimizing a single score.

**Artifacts:**
- artifacts/bias_correction_v*.json
- artifacts/error_analysis_bias_comparison_v*.csv
- artifacts/error_analysis_groupwise_comparison_v*.csv
- visuals/calibration_deciles_bias_corrected_v*.png
- visuals/calibration_deciles_groupwise_v*.png

### Prediction intervals (recommended output)

Single euro point predictions are misleading for heavy-tailed targets.

Instead, the project computes prediction intervals using empirical residual quantiles from training data.

**Implemented:**
- Central 80% prediction interval (10–90%)
- Empirical coverage validation on future test data

**Example (v3):**
- Target coverage: 80%
- Empirical coverage on test: ~81%

**Artifacts:**
- artifacts/prediction_intervals_v*.csv
- artifacts/prediction_interval_summary_v*.json
- visuals/prediction_interval_distribution_v*.png

This preserves median accuracy while making uncertainty explicit.

## Results summary (current)

- Ridge regression significantly outperforms naïve baselines
- Performance features contain signal, but context features explain a large share of variance
- Nationality contributes some signal but is not dominant
- Log-target bias is measurable and quantifiable
- Prediction intervals show strong empirical calibration on future data

The model is intentionally simple. The focus is correctness, interpretability, and evaluation — not benchmark chasing.

## Reproducibility

**Install dependencies:**
```
pip install -r requirements.txt
```

**(Optional) Build local DuckDB for fast analytics:**
```
python scripts/04_build_duckdb.py
```

**Train the model and generate artifacts:**
```
python scripts/03_train_market_value.py
```

**Run diagnostics:**
```
python scripts/06_error_analysis.py
```

**Run bias correction experiments:**
```
python scripts/07_bias_correction.py
python scripts/08_groupwise_bias_correction.py
```

**Generate prediction intervals:**
```
python scripts/09_prediction_intervals.py
```

## Project structure

```
footytrackr/
├── data/
│   ├── raw/ (ignored in Git)
│   ├── processed/ (ignored in Git)
│   ├── features/
│   └── db/ (DuckDB, ignored in Git)
├── scripts/
│   ├── 01_profile_raw.py
│   ├── 02_build_value_features_v*.py
│   ├── 03_train_market_value.py
│   ├── 04_build_duckdb.py
│   ├── 06_error_analysis.py
│   ├── 07_bias_correction.py
│   ├── 08_groupwise_bias_correction.py
│   └── 09_prediction_intervals.py
├── artifacts/
├── notebooks/
├── visuals/
├── requirements.txt
└── README.md
```

## Tools used

- **Python**
  - pandas, numpy
  - scikit-learn
  - joblib
  - duckdb
  - matplotlib
- **Jupyter notebooks**
- **VS Code**

No AutoML. No deep learning frameworks. No black boxes.

## Why this project exists

This project is used to:
- Practice data cleaning on imperfect real-world data
- Design evolving feature pipelines
- Evaluate models with leakage-safe splits
- Reason about model behavior and uncertainty
- Build artifact-driven, production-like workflows
- Communicate results clearly

It is deliberately opinionated and iterative.

## Disclaimer

Transfermarkt market values are estimates and reflect market perception rather than objective player skill.
This project analyzes patterns in the data, not absolute player quality.