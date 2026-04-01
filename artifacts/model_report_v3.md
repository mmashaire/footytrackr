# Footytrackr model report

Model version: `v3`

## Overall snapshot

- Test rows: 99,567
- Cutoff date: `2022-05-18`
- MAE: EUR 2.34M
- Median absolute error: EUR 385K
- Mean signed error: EUR -1.65M (underprediction)
- Interval coverage: 80.8% vs target 80.0%

## Risk flags

- Interval coverage is close to target, which is a good sign for calibration.
- Average signed error stays negative, so the model is still underpricing players overall.
- Highest position-level error is in Other profiles, so that subgroup deserves a closer look.
- League breakdown is not yet very informative because the current artifact is mostly tagged as 'Unknown'.

## Highest-error positions

- **Other** — MAE EUR 2.66M across 28632 rows
- **MID** — MAE EUR 2.62M across 28476 rows
- **DEF** — MAE EUR 2.21M across 31914 rows

## Highest-error leagues

- League breakdown is currently too sparse to rank meaningfully.
