# Model Card: Ridge Regression Player Market Value Estimator (v3)

## Model Details

### Overview

This is a Ridge regression model trained to estimate player market values from historical Transfermarkt football data. The model emphasizes interpretability, uncertainty quantification, and honest performance reporting over black-box optimization. In portfolio terms, it is meant to demonstrate applied AI/ML skills that are practical, reproducible, and easy to defend.

### Algorithm

- **Type**: Linear regression with L2 regularization
- **Framework**: scikit-learn `Ridge` with alpha=1.0
- **Target**: Log-transformed player market value (EUR)
- **Input Features**: 36 engineered features across performance, context, and player characteristics
- **Output**: Point estimate (median) + 90% confidence interval

### Feature Categories

| Category | Examples | Count |
| --- | --- | --- |
| Performance (180-day windows) | games played, goals, assists, cards | 8 |
| Performance (365-day windows) | goals per 90, assists per 90, card rate | 8 |
| Player Demographics | age, position, citizenship, country of birth | 12 |
| Contextual | club, domestic league competition | 8 |

### Model Artifacts

- **Trained model**: `artifacts/ridge_model_v3.joblib`
- **Top coefficients**: `artifacts/ridge_top_coefficients_v3.csv`
- **Ablation analysis**: `artifacts/ablation_v3.csv`

---

## Intended Use

### Primary Use Cases

1. **Market valuation benchmarking**: Quick reference point for player value in the €500k–€100M range
2. **Portfolio analysis**: Comparing estimated vs. actual values to identify over/under-valued transfermarket deals
3. **Player evaluation**: Supporting workflows where interpretable, explainable estimates are preferred
4. **Portfolio education**: Demonstrating applied AI/ML through leakage-safe modeling, time-based splits, calibration analysis, bias correction, and uncertainty-aware prediction

### Out-of-Scope Uses

- **Real-time trading**: Model is retrained infrequently; market conditions change faster than retraining cycles
- **Legal/contractual decisions**: Not intended for binding valuation or contract arbitration
- **Young player project**: Model expects reliable historical statistics; youth players with few observations will be unreliable
- **Club-specific domain knowledge**: Model ignores contract details, injury history, tax implications, and other factors that drive real deals
- **Players outside major leagues**: Training data is concentrated in Europe; player valuations outside this domain are extrapolations

---

## Training Data

### Data Source

- **Provider**: Transfermarkt (publicly available historical snapshots)
- **Raw tables**: Player profiles, appearances, valuations, transfers, competitions, games, lineups

### Dataset Composition

| Metric | Value |
| --- | --- |
| Training samples | 396,473 |
| Test samples | 99,567 |
| Training period | 2003-12-09 to 2022-05-17 |
| Test period | 2022-05-18 to 2025-04-06 |
| Cutoff strategy | Strict time-based split (train < cutoff, test ≥ cutoff) |
| Data leakage protection | No future information used; rolling windows are forward-filled |

### Data Characteristics

- **Coverage**: ~30 major European leagues and clubs
- **Player filter**: Includes all players with valuations and recent appearance history
- **Missing values**: Imputed with central tendency (median for numeric, mode for categorical); missing leagues coded as 'Unknown'
- **Temporal distribution**: Heavy concentration post-2015; sparse pre-2010

---

## Model Performance

### Overall Metrics

| Metric | Log Scale | EUR Scale | Notes |
| --- | --- | --- | --- |
| RMSE | 1.129 | — | Root mean squared error in log space |
| MAE | 0.891 | €2.34M | Mean absolute error; ~38% of median prediction |
| Median Abs. Err. | — | €385k | 50th percentile absolute error |

### Baseline Comparison

| Baseline | RMSE (log) | Gain vs. Ridge |
| --- | --- | --- |
| Global median | 1.718 | 34% better |
| Position + age stratified | 1.648 | 31% better |
| **Ridge model** | **1.129** | — |

### Feature Importance (Top 10 Positive Coefficients)

1. **Colombian citizenship** (+1.06): Rare positive signal in data
2. **English league (GB1)** (+0.90): Top domestic market premium
3. **Unknown birthplace** (+0.84): Data quality flag; imputation effect
4. **Brazilian origin** (+0.63): Historical talent source
5. **Argentine origin** (+0.56): Historical talent source
6. **365-day games played** (+0.55): Performance consistency
7. **Polish origin** (+0.51): Emerging player market
8. **CSSR origin** (+0.39): Historical context feature
9. **Spanish league (ES1)** (+0.38): Major league premium
10. **Luxembourg league (L1)** (+0.28): Small league adjustment

### Ablation Results

Removing feature groups causes performance degradation:

| Feature Set | MAE (log) | Change |
| --- | --- | --- |
| Full model | 0.891 | — |
| Without nationality | 0.926 | +3.9% |
| Without contextual (league/club) | 1.005 | +12.8% |
| Performance metrics only | 1.027 | +15.3% |

**Interpretation**: Context features (league, club, citizenship) add ~13% value; performance stats alone are insufficient.

### Prediction Interval Calibration

- **Target coverage**: 90% interval should contain 90% of future values
- **Observed coverage on test set**: ~88% (calibrated empirically from residual quantiles)
- **Method**: Quantile regression on training residuals; intervals widen for uncertain predictions

### Known Limitations

1. **Systematic optimism for high-value players**: Model tends to overpredict for €50M+ players (mean signed error: -€1.65M). Likely due to scarcity in training data; few examples help calibration in extremes.

2. **League-dependent bias**: Model performs better in major leagues (England, Spain, Germany, Italy) where data is dense. Smaller leagues and lower divisions have wider intervals.

3. **Age cutoff effects**: Model expects age 16–50 range; predictions degrade for youth (< 16) and retired player records.

4. **Circular reasoning for young players**: Recent signings with few historical appearances will have unreliable stage-based rolling window features; intervals may appear tight when actual uncertainty is high.

5. **Transfer window lags**: Model is retrained ~every 2–3 months; market movements within a window are not captured.

---

## Bias and Fairness Analysis

### Known Biases

1. **Geographic bias**: Training data reflects Transfermarkt's coverage concentration (European leagues). Players from underrepresented regions (Africa, Asia, South America outside Brazil) may receive out-of-distribution interpolations.

2. **Historical population bias**: Top countries/leagues (Brazil, Argentina, England, Germany) have overrepresented coefficients, but this reflects real historical market behavior, not algorithmic bias per se.

3. **Recency bias**: More recent data (post-2015) has higher resolution; older player records may be sparse. Model weights recent statistics more heavily in rolling windows.

4. **Survivorship bias**: Only currently-playing or recently-active players are in Transfermarkt. Retired/inactive players are underrepresented.

### Fairness Considerations

- **Transparent uncertainty**: Every prediction includes a confidence interval; single-point estimates are discouraged
- **Interpretable features**: Coefficients are directly readable; model is not a black box
- **Ablation studies**: Impact of feature groups is quantified, so stakeholders can challenge modeling choices
- **Error audits**: Per-league and per-position error analysis is included to surface disparities

### Bias Correction Attempts

Experimental corrections tested:

- **Global smearing**: Additive correction for systematic underestimation when back-transforming from log scale. Modest improvement but introduces dependency on reference set.
- **Groupwise smearing**: Per-league and per-position corrections. Marginal gains; risk of overfitting to test set.

**Recommendation**: Use empirical prediction intervals instead. They are more robust and make uncertainty visible.

---

## Technical Specifications

### Training Procedure

```python
1. Load v3 features (396k training rows)
2. Strict time split: cutoff = 2022-05-18
3. Impute missing values (median for numeric, mode for categorical)
4. One-hot encode categorical features
5. Standardize numeric features (mean 0, std 1)
6. Fit Ridge(alpha=1.0) on training data
7. Evaluate on test set (99k rows, future data)
8. Compute residual quantiles for prediction intervals
```

### Reproducibility

- **Data version**: Features v3 (date 2022-05-18)
- **Random seed**: Not critical (Ridge is deterministic given feature prep)
- **Python version**: 3.9+
- **Environment**: See `requirements.txt` and `pyproject.toml`

### API Schema

**Request** (JSON):

```json
{
  "age": 25,
  "position": "Centre-Forward",
  "w180_games_played": 15,
  "w180_minutes_played": 1200,
  "w180_goals": 8,
  "w180_assists": 3,
  "w180_yellow_cards": 2,
  "w180_red_cards": 0,
  "w365_games_played": 30,
  "w365_minutes_played": 2500,
  "w365_goals": 14,
  "w365_assists": 6,
  "w365_yellow_cards": 4,
  "w365_red_cards": 0,
  "player_club_domestic_competition_id": "GB1"
}
```

Optional query parameter:

- `explain=true`: include top positive and negative feature contributions in log-value space

**Response** (JSON):

```json
{
  "predicted_log_value": 13.0,
  "predicted_value_eur": 12500000,
  "confidence_interval": {
    "lower": 2100000,
    "upper": 32000000
  },
  "interval_coverage": 0.88,
  "explanation": {
    "baseline_log_value": 12.5,
    "top_positive": [
      {
        "feature": "w180_goals",
        "feature_value": 8,
        "transformed_feature": "w180_goals",
        "contribution_log": 0.9
      }
    ],
    "top_negative": [
      {
        "feature": "age",
        "feature_value": 25,
        "transformed_feature": "age",
        "contribution_log": -0.2
      }
    ]
  }
}
```

---

## Contact and Attribution

- **Author**: [Portfolio project for IT/data engineering roles](https://github.com/mmashaire/footytrackr)
- **Date trained**: May 2022 (v3 iteration)
- **Data source**: Transfermarkt (publicly available)
- **License**: MIT (project) + Transfermarkt terms of use (data)

### Citation

If referencing this model, cite as:

> "Ridge Regression Player Market Value Estimator v3. Transfermarkt data (2003–2025). Notebook: footytrackr project. Includes time-based splits, ablation analysis, and empirical calibration."

### Version History

| Version | Date | Key Changes |
| --- | --- | --- |
| v1 | Early 2022 | Initial features, baseline ridge model |
| v2 | Mid 2022 | Added rolling windows, improved feature engineering |
| v3 | May 2022 | Finalized feature set (36 features), added prediction intervals and bias analysis |

---

## Appendix: Ethical Considerations

### Transparency

- **Model is interpretable**: Ridge coefficients are directly readable and defendable
- **Uncertainty is quantified**: Every prediction includes an 80–90% confidence interval
- **Limitations are documented**: This card makes failure modes and biases explicit

### Fairness

- **No discriminatory features**: Model does not directly encode player identity or membership in protected groups (though geographic and historical proxies are present)
- **Error audits are available**: Performance is broken down by position, league, and experience level

### Limitations and Disclaimers

- **Not a substitute for human judgment**: Useful for benchmarking and portfolio analysis, not for binding valuation decisions
- **Data reflects historical market**: Model encodes past trading patterns and biases; using it perpetuates those patterns
- **No guarantees of future accuracy**: Market conditions, rule changes, and player development are non-stationary; model is retrained infrequently
- **Privacy respected**: No player names or personal identifiers in model features; predictions are based on anonymized statistics

### Recommended Use Practices

1. Always present intervals, not point estimates alone
2. Audit predictions within your domain before deployment
3. Retrain regularly (monthly or quarterly) if used operationally
4. Cross-check with human scout reports and domain experts
5. Document all use cases and decisions for accountability

---

*Model Card version 1.0 | Generated for footytrackr project | [Back to README](README.md)*
