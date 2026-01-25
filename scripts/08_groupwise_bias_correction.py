from pathlib import Path
import json

import duckdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import load

MODEL_VERSION = "v3"
CUTOFF_DATE = "2022-05-18"
GROUP_COL = "player_club_domestic_competition_id"

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


def eval_errors(actual, pred):
    abs_err = (pred - actual).abs()
    signed_err = pred - actual
    return {
        "mae_eur": float(abs_err.mean()),
        "median_abs_err_eur": float(abs_err.median()),
        "mean_signed_err_eur": float(signed_err.mean()),
    }


def calibration(actual, pred):
    df = pd.DataFrame({"actual": actual, "pred": pred}).sort_values("pred")
    df["decile"] = pd.qcut(df["pred"], 10, labels=False, duplicates="drop")
    return (
        df.groupby("decile")
        .agg(
            pred_mean=("pred", "mean"),
            actual_mean=("actual", "mean"),
            n=("actual", "count"),
        )
        .reset_index()
    )


def main():
    ensure_dirs()

    if not MODEL_PATH.exists():
        raise FileNotFoundError("Train model first (03_train_market_value.py).")

    train, test = load_split()
    model = load(MODEL_PATH)

    # --------------------
    # Train: group smear
    # --------------------
    train = train.dropna(subset=["target_log"]).copy()
    X_train = train[feature_cols(train)].fillna(0)
    y_train = train["target_log"]

    train["pred_log"] = model.predict(X_train)
    train["resid_log"] = y_train - train["pred_log"]

    # Compute smearing factor per group
    smear_table = (
        train.groupby(GROUP_COL)
        .apply(lambda g: float(np.mean(np.exp(g["resid_log"]))))
        .rename("smearing_factor")
        .reset_index()
    )

    # Fallback for unseen groups
    global_smear = float(np.mean(np.exp(train["resid_log"])))

    smear_table.to_csv(
        ARTIFACTS_DIR / f"groupwise_smearing_{MODEL_VERSION}.csv",
        index=False,
    )

    # --------------------
    # Test predictions
    # --------------------
    test = test.dropna(subset=["target_log"]).copy()
    X_test = test[feature_cols(test)].fillna(0)

    test["pred_log"] = model.predict(X_test)
    test["pred_eur"] = safe_expm1(test["pred_log"])
    test["actual_eur"] = test["market_value_in_eur"].clip(lower=0)

    # Merge group smearing
    test = test.merge(smear_table, on=GROUP_COL, how="left")
    test["smearing_factor"] = test["smearing_factor"].fillna(global_smear)

    test["pred_eur_groupwise"] = (
        test["pred_eur"] * test["smearing_factor"]
    ).clip(lower=0)

    # --------------------
    # Evaluation
    # --------------------
    base_metrics = eval_errors(test["actual_eur"], test["pred_eur"])
    group_metrics = eval_errors(
        test["actual_eur"], test["pred_eur_groupwise"]
    )

    summary = {
        "baseline": base_metrics,
        "groupwise_bias_corrected": group_metrics,
        "global_fallback_smearing": global_smear,
        "n_test": int(len(test)),
        "group_col": GROUP_COL,
    }

    with open(
        ARTIFACTS_DIR / f"error_analysis_groupwise_bias_{MODEL_VERSION}.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(summary, f, indent=2)

    comp = pd.DataFrame(
        [
            {"variant": "baseline", **base_metrics},
            {"variant": "groupwise_bias_corrected", **group_metrics},
        ]
    )
    comp.to_csv(
        ARTIFACTS_DIR / f"error_analysis_groupwise_comparison_{MODEL_VERSION}.csv",
        index=False,
    )

    # Calibration plot
    cal = calibration(test["actual_eur"], test["pred_eur_groupwise"])
    cal.to_csv(
        ARTIFACTS_DIR / f"calibration_deciles_groupwise_{MODEL_VERSION}.csv",
        index=False,
    )

    plt.figure()
    plt.plot(cal["pred_mean"], cal["actual_mean"], marker="o")
    plt.title("Calibration (groupwise bias-corrected)")
    plt.xlabel("Mean predicted EUR")
    plt.ylabel("Mean actual EUR")
    plt.tight_layout()
    plt.savefig(
        VISUALS_DIR / f"calibration_deciles_groupwise_{MODEL_VERSION}.png",
        dpi=160,
    )
    plt.close()

    print("✅ Groupwise bias correction complete.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
