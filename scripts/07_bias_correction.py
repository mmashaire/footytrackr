from pathlib import Path
import json

import duckdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import load

MODEL_VERSION = "v3"
CUTOFF_DATE = "2022-05-18"

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "db" / "footytrackr.duckdb"
ARTIFACTS_DIR = ROOT / "artifacts"
VISUALS_DIR = ROOT / "visuals"
MODEL_PATH = ARTIFACTS_DIR / f"ridge_model_{MODEL_VERSION}.joblib"


def ensure_dirs() -> None:
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    VISUALS_DIR.mkdir(exist_ok=True)


def load_split_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not DB_PATH.exists():
        raise FileNotFoundError("DuckDB database not found. Run scripts/04_build_duckdb.py first.")

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


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    drop_cols = {
        "market_value_in_eur",
        "target_log",
        "date",
        "player_id",
        "name",
        "full_name",
    }
    return [c for c in df.columns if c not in drop_cols]


def safe_expm1(series: pd.Series) -> pd.Series:
    # Avoid weird negatives and keep the output stable.
    vals = np.expm1(series.to_numpy())
    return pd.Series(vals).clip(lower=0)


def eval_errors(actual_eur: pd.Series, pred_eur: pd.Series) -> dict:
    abs_err = (pred_eur - actual_eur).abs()
    signed_err = pred_eur - actual_eur

    return {
        "mae_eur": float(abs_err.mean()),
        "median_abs_err_eur": float(abs_err.median()),
        "mean_signed_err_eur": float(signed_err.mean()),
    }


def calibration_deciles(actual_eur: pd.Series, pred_eur: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({"actual_eur": actual_eur, "pred_eur": pred_eur}).copy()
    df = df.sort_values("pred_eur")
    df["decile"] = pd.qcut(df["pred_eur"], 10, labels=False, duplicates="drop")

    cal = (
        df.groupby("decile")
        .agg(
            pred_mean=("pred_eur", "mean"),
            actual_mean=("actual_eur", "mean"),
            n=("actual_eur", "count"),
        )
        .reset_index()
    )
    return cal


def plot_calibration(cal: pd.DataFrame, outpath: Path, title: str) -> None:
    plt.figure()
    plt.plot(cal["pred_mean"], cal["actual_mean"], marker="o")
    plt.title(title)
    plt.xlabel("Mean predicted EUR")
    plt.ylabel("Mean actual EUR")
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def main() -> None:
    ensure_dirs()

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}. Run scripts/03_train_market_value.py first.")

    train_df, test_df = load_split_frames()

    required = ["target_log", "market_value_in_eur"]
    for col in required:
        if col not in train_df.columns or col not in test_df.columns:
            raise ValueError(f"Missing required column '{col}' in DuckDB table.")

    model = load(MODEL_PATH)

    # --- Train residuals (log space) ---
    train_df = train_df.dropna(subset=["target_log"]).copy()
    X_train = train_df[get_feature_cols(train_df)].fillna(0)
    y_train_log = train_df["target_log"].astype(float)

    yhat_train_log = model.predict(X_train)
    resid_log = y_train_log.to_numpy() - yhat_train_log

    # Duan smearing estimator: S = mean(exp(residual))
    smear = float(np.mean(np.exp(resid_log)))

    bias_info = {
        "model_version": MODEL_VERSION,
        "cutoff_date": CUTOFF_DATE,
        "n_train": int(len(train_df)),
        "smearing_factor": smear,
    }
    with open(ARTIFACTS_DIR / f"bias_correction_{MODEL_VERSION}.json", "w", encoding="utf-8") as f:
        json.dump(bias_info, f, indent=2)

    # --- Test predictions (before/after) ---
    test_df = test_df.dropna(subset=["target_log"]).copy()
    X_test = test_df[get_feature_cols(test_df)].fillna(0)

    pred_log = pd.Series(model.predict(X_test))
    pred_eur = safe_expm1(pred_log)

    # Bias-corrected EUR prediction:
    # Apply smearing to expm1 output (approx; works well for heavy tails)
    pred_eur_bc = (pred_eur * smear).clip(lower=0)

    actual_eur = test_df["market_value_in_eur"].clip(lower=0)

    base_metrics = eval_errors(actual_eur, pred_eur)
    bc_metrics = eval_errors(actual_eur, pred_eur_bc)

    out = {
        "baseline": base_metrics,
        "bias_corrected": bc_metrics,
        "smearing_factor": smear,
        "n_test": int(len(test_df)),
    }
    with open(
        ARTIFACTS_DIR / f"error_analysis_bias_corrected_{MODEL_VERSION}.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(out, f, indent=2)

    comp = pd.DataFrame(
        [
            {"variant": "baseline", **base_metrics},
            {"variant": "bias_corrected", **bc_metrics},
        ]
    )
    comp.to_csv(ARTIFACTS_DIR / f"error_analysis_bias_comparison_{MODEL_VERSION}.csv", index=False)

    # Calibration plot for bias-corrected predictions (helps tell the story)
    cal_bc = calibration_deciles(actual_eur, pred_eur_bc)
    cal_bc.to_csv(ARTIFACTS_DIR / f"calibration_deciles_bias_corrected_{MODEL_VERSION}.csv", index=False)
    plot_calibration(
        cal_bc,
        VISUALS_DIR / f"calibration_deciles_bias_corrected_{MODEL_VERSION}.png",
        title="Calibration (bias-corrected): mean predicted EUR vs mean actual EUR",
    )

    print("✅ Bias correction complete.")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
