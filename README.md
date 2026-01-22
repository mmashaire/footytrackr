# footytrackr

`footytrackr` is a football data project built around historical Transfermarkt data.  
The goal is to practice realistic data engineering and applied data science workflows using a large, messy, real-world dataset.

The project focuses on:
- building reproducible feature pipelines
- training and evaluating simple, interpretable models
- understanding what actually drives player market value through analysis and ablation studies

This is not a tutorial project. It is intentionally exploratory and iterative.

---

## Dataset

The data originates from Transfermarkt and includes:
- player profiles and market values
- match appearances and minutes played
- goals, assists, cards, and playing time
- club, competition, and league context
- transfer history

The raw dataset contains hundreds of thousands of rows spanning multiple decades and leagues.

---

## What is implemented

### Feature pipeline
Raw CSVs are transformed into model-ready feature tables using versioned scripts:

- Rolling performance aggregates over 180 and 365 days
- Per-90 performance rates (goals, assists, cards)
- Age, position, physical attributes
- League and club context
- Categorical grouping to control feature cardinality

Feature versions are saved explicitly (`v1`, `v2`, `v3`) to allow comparisons.

---

### Market value model
A linear Ridge regression model is trained to predict log-transformed market value.

Key design choices:
- strict time-based train/test split to prevent leakage
- no direct identifiers or raw euro values used as features
- median and position+age baselines for comparison
- model interpretability preserved through linear coefficients

Artifacts saved per run:
- trained model (`.joblib`)
- evaluation metrics (`.json`)
- top positive and negative coefficients (`.csv`)

---

### Ablation study
An ablation study is used to understand feature reliance.

The same model is trained under different feature restrictions:
- full feature set
- without nationality features
- without nationality and league context
- performance-only features

This allows direct measurement of how much each feature group contributes to predictive performance.

Results are saved as a separate ablation table for comparison.

---

## Results summary (current)

- The Ridge model significantly outperforms naïve baselines.
- Performance metrics alone capture meaningful signal.
- League context explains a large portion of market value variation.
- Nationality contributes moderately but is not the primary driver.

The model is intentionally simple. The focus is on understanding the data and feature effects, not maximizing benchmark scores.

---

## Project structure

footytrackr/
├── data/
│ ├── raw/ # original CSVs (ignored in Git)
│ └── features/ # versioned feature tables
├── scripts/
│ ├── 01_profile_raw.py
│ ├── 02_build_value_features_v*.py
│ └── 03_train_market_value.py
├── artifacts/
│ ├── metrics_v*.json
│ ├── ridge_model_v*.joblib
│ ├── ridge_top_coefficients_v*.csv
│ └── ablation_v*.csv
├── notebooks/ # exploratory analysis
└── visuals/ # saved plots and figures
---

## Tools used

- Python
- pandas, numpy
- scikit-learn
- joblib
- Jupyter notebooks
- VS Code

No AutoML, no deep learning frameworks, no black boxes.

---

## Why this project exists

This project is used as a sandbox to:
- practice data cleaning on imperfect data
- design feature pipelines that evolve over time
- evaluate models properly using time-aware splits
- reason about feature importance and model behavior
- build artifacts that resemble real production workflows

It is deliberately opinionated and iterative.

---

## Next steps (optional)

- calibration analysis and prediction error plots
- position-specific models
- cross-league generalization checks
- lightweight database layer (DuckDB)
- simple dashboard for inspection

---

## Disclaimer

Transfermarkt market values are estimates and reflect market perception rather than objective player skill.  
This project analyzes patterns in the data, not absolute player quality.


