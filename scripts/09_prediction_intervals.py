from pathlib import Path
import json

import duckdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import load

MODEL_VERSION = "v3"
CUTOFF_DATE = "2022-05-18"

# Interval definition (central 80%)
LOW_Q = 0.10
HIGH_Q = 0.90

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "db" / "footytrackr.duckdb"
ARTIFACTS_DIR = ROOT / "artifacts"
VISUALS_DIR = ROOT / "visuals"
MODEL_PATH = ARTIFACTS_DIR / f"ridge_model_{MODEL_VERSION}.joblib"


def ensure_dirs():
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    VISUALS_DIR.mkdir(exist_ok=True)


def load_split():
    con = duckdb.connect(str(DB_PATH), read_only=True)

    train = con.execute(
        f"""
        SELECT *
        FROM v_model_rows
        WHERE date < '{CUTOFF_DATE}'
        """
    ).df()

    test = con.execute(
        f"""
        SELECT *
        FROM v_model_rows
        WHERE date >= '{CUTOFF_DATE}'
        """
    ).df()

    con.close()
    return train, test


def feature_cols(df):
    drop = {
        "market_value_in_eur",
        "target_log",
        "date",
        "player_id",
        "name",
        "full_name",
    }
    return [c for c in df.columns if c not in drop]


def safe_expm1(x):
    return pd.Series(np.expm1(x)).clip(lower=0)


def main():
    ensure_dirs()

    if not MODEL_PATH.exists():
        raise FileNotFoundError("Train model first (03_train_market_value.py).")

    model = load(MODEL_PATH)
    train, test = load_split()

    # -------------------------
    # Train residuals (log)
    # -------------------------
    train = train.dropna(subset=["target_log"]).copy()
    X_train = train[feature_cols(train)].fillna(0)
    y_train = train["target_log"].astype(float)

    train["pred_log"] = model.predict(X_train)
    train["resid_log"] = y_train - train["pred_log"]

    q_low = float(train["resid_log"].quantile(LOW_Q))
    q_med = float(train["resid_log"].quantile(0.50))
    q_high = float(train["resid_log"].quantile(HIGH_Q))

    # -------------------------
    # Test predictions
    # -------------------------
    test = test.dropna(subset=["target_log"]).copy()
    X_test = test[feature_cols(test)].fillna(0)

    test["pred_log"] = model.predict(X_test)

    test["pred_log_lower"] = test["pred_log"] + q_low
    test["pred_log_median"] = test["pred_log"] + q_med
    test["pred_log_upper"] = test["pred_log"] + q_high

    test["pred_eur_lower"] = safe_expm1(test["pred_log_lower"])
    test["pred_eur_median"] = safe_expm1(test["pred_log_median"])
    test["pred_eur_upper"] = safe_expm1(test["pred_log_upper"])
    test["actual_eur"] = test["market_value_in_eur"].clip(lower=0)

    # -------------------------
    # Coverage check
    # -------------------------
    inside = (
        (test["actual_eur"] >= test["pred_eur_lower"])
        & (test["actual_eur"] <= test["pred_eur_upper"])
    )
    coverage = float(inside.mean())

    summary = {
        "model_version": MODEL_VERSION,
        "interval": f"{int(LOW_Q*100)}–{int(HIGH_Q*100)}%",
        "target_coverage": HIGH_Q - LOW_Q,
        "empirical_coverage": coverage,
        "n_test": int(len(test)),
        "residual_quantiles_log": {
            "q10": q_low,
            "q50": q_med,
            "q90": q_high,
        },
    }

    with open(
        ARTIFACTS_DIR / f"prediction_interval_summary_{MODEL_VERSION}.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(summary, f, indent=2)

    # Save interval predictions (lightweight columns only)
    cols_keep = [
        "date",
        "player_id" if "player_id" in test.columns else None,
        "pred_eur_lower",
        "pred_eur_median",
        "pred_eur_upper",
        "actual_eur",
    ]
    cols_keep = [c for c in cols_keep if c is not None]

    test[cols_keep].to_csv(
        ARTIFACTS_DIR / f"prediction_intervals_{MODEL_VERSION}.csv",
        index=False,
    )

    # -------------------------
    # Coverage plot
    # -------------------------
    plt.figure()
    plt.hist(
        test["actual_eur"],
        bins=50,
        alpha=0.6,
        label="Actual EUR",
    )
    plt.hist(
        test["pred_eur_median"],
        bins=50,
        alpha=0.6,
        label="Predicted median EUR",
    )
    plt.legend()
    plt.title("Distribution: actual vs predicted (median)")
    plt.tight_layout()
    plt.savefig(
        VISUALS_DIR / f"prediction_interval_distribution_{MODEL_VERSION}.png",
        dpi=160,
    )
    plt.close()

    print("✅ Prediction intervals complete.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
