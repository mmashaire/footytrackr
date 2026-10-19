# Footytrackr

[![CI](https://github.com/mmashaire/footytrackr/actions/workflows/ci.yml/badge.svg)](https://github.com/mmashaire/footytrackr/actions/workflows/ci.yml)

Footytrackr is a football data project that estimates a player's Transfermarkt
market value from their recent performance and player profile. It shows the
full path from data preparation to model evaluation and a small prediction API.

The estimates are for learning and comparison. They are not transfer fees,
financial advice, or a replacement for a scout's judgment.

## What the project does

- Builds player features from historical data, including statistics from the
  previous 180 and 365 days.
- Trains an interpretable Ridge regression model.
- Tests the model on later dates than the training data.
- Reports prediction errors and an 80% prediction interval.
- Provides a command-line interface and a FastAPI service.
- Compares similar players using their latest available data.

## Model results

These results come from the saved v3 evaluation. The model was trained on
valuations from 2003-12-09 through 2022-05-17 and tested on later valuations
through 2025-04-06.

| Measure | Result |
| --- | ---: |
| Test rows | 99,567 |
| Mean absolute error | €2.34M |
| Median absolute error | €385K |
| Log-scale RMSE | 1.129 (global median baseline: 1.718) |
| 80% prediction interval coverage | 80.8% |

The large difference between mean and median error reflects a long-tailed
valuation range: a small number of very expensive players can strongly affect
the mean error. The model also underpredicts on average in the saved evaluation.
See [`artifacts/model_report_v3.md`](artifacts/model_report_v3.md) and the
[Model Card](MODELCARD.md) for more detail.

These are historical results, not a promise of future accuracy. The data ends
on 2025-04-06, and market values are estimates rather than confirmed transfer
prices.

## How the estimate works

The model uses 36 features, including age, position, league, and recent
appearance statistics. It predicts the log of market value, then converts the
result back to euros.

The API also returns an **80% prediction interval**. In the held-out test data,
about 80.8% of values fell inside that range. This is an overall test result;
it does not mean every player's interval has the same accuracy.

## Try the project

### Install and run tests

Use Python 3.10 or 3.11, which are the versions tested by CI:

```bash
python -m pip install -e ".[dev,api]"
python -m pytest tests/ -q
```

### Before making predictions

The source code, tests, examples, and small evaluation reports are on GitHub.
The raw data and trained model file are not. They are kept out of Git, so a
fresh clone can run the tests but cannot make predictions until you supply the
data and model artifacts locally.

For predictions, the API needs:

- `artifacts/ridge_model_v3.joblib`
- `artifacts/prediction_interval_summary_v3.json`

The model file is created by the training workflow. To rebuild the model, first
place the required source CSVs in `data/raw/`; those data files are not included
in this repository. Check the data provider's terms before using or sharing
those files.

Once the model artifacts are available, check readiness and make a prediction:

```bash
footytrackr health
footytrackr predict --input-file examples/young_striker_gb1.json --explain
```

To include a short scouting summary:

```bash
footytrackr predict \
  --input-file examples/young_striker_gb1.json \
  --explain \
  --scout-assistant
```

## Run the API

With the model artifacts in place, start the service:

```bash
footytrackr api --host 127.0.0.1 --port 8000
```

The interactive API guide is at `http://127.0.0.1:8000/docs`.

Main endpoints:

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Check that the model and prediction interval are ready. |
| `POST /predict` | Estimate a player's value from a JSON feature set. |
| `GET /scout/similar` | Compare a player with similar players in the same league. |

Example request:

```bash
curl -X POST "http://127.0.0.1:8000/predict?explain=true" \
  -H "Content-Type: application/json" \
  -d @examples/young_striker_gb1.json
```

### Similar-player comparisons

For example, compare Lionel Messi with up to five players:

```bash
curl "http://127.0.0.1:8000/scout/similar?player=Lionel%20Messi&n=5"
```

The endpoint uses the latest available feature row for each player, then
compares players with the same position and league and within two years of age.
It includes the date for each row because players' records may come from
different dates. Results are ranked by the difference between the model estimate
and the recorded market value. A positive difference is a reason to investigate,
not proof that a player is undervalued.

Scouting also needs the local feature file
`data/features/player_value_features_v3.csv` and player names in
`data/processed/players_clean.csv`.

## Data and training workflow

The scripts expect local input files under `data/`. That folder is not tracked
by Git. The feature-building and model-training commands are:

```bash
footytrackr build-features --version v3
footytrackr train
```

The repository also includes scripts for error analysis, bias checks, prediction
intervals, and reports. Run them with the matching `footytrackr` subcommands;
they rely on the local feature data and artifacts. For example:

```bash
footytrackr error-analysis
footytrackr prediction-intervals
footytrackr model-report
```

## Project files

```text
footytrackr/   API, command-line interface, and report generation
scripts/       Data preparation, feature building, training, and analysis
tests/         Automated tests
examples/      Example prediction requests
artifacts/     Saved metrics, reports, and diagnostic results
data/          Local source and processed data (not tracked by Git)
MODELCARD.md   Model details, intended use, and limitations
```

## Checks and documentation

GitHub Actions runs formatting, import-order, type, and lint checks, then runs
the test suite on Python 3.10 and 3.11.

- [Model Card](MODELCARD.md)
- [Contributing guide](CONTRIBUTING.md)
- [CI workflow](.github/workflows/ci.yml)

## Disclaimer

Transfermarkt market values are estimates, not ground truth. This model learns
patterns in those estimates. Use its predictions as one source of information,
and consider their uncertainty and the data's age.
