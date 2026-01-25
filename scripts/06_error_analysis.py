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


def load_test_frame() -> pd.DataFrame:
    if not DB_PATH.exists():
        raise FileNotFoundError("DuckDB database not found. Run scripts/04_build_duckdb.py first.")

    con = duckdb.connect(str(DB_PATH), read_only=True)

    df = con.execute(
        f"""
        SELECT *
        FROM v_model_rows
        WHERE date >= '{CUTOFF_DATE}'
        """
    ).df()

    con.close()
    return df


def make_position_bucket(pos: str) -> str:
    if pos is None or (isinstance(pos, float) and np.isnan(pos)):
        return "Unknown"

    p = str(pos).lower()

    if any(k in p for k in ["goalkeeper", "keeper", "gk"]):
        return "GK"
    if any(
        k in p
        for k in [
            "defender",
            "back",
            "centre-back",
            "center-back",
            "cb",
            "lb",
            "rb",
            "lwb",
            "rwb",
            "wing-back",
        ]
    ):
        return "DEF"
    if any(k in p for k in ["midfield", "midfielder", "dm", "cm", "am", "wing", "wide"]):
        return "MID"
    if any(
        k in p
        for k in [
            "forward",
            "striker",
            "second striker",
            "centre-forward",
            "center-forward",
            "cf",
            "st",
            "winger",
        ]
    ):
        return "FWD"

    return "Other"


def plot_residual_hist(df: pd.DataFrame, outpath: Path) -> None:
    plt.figure()
    plt.hist(df["resid_log"], bins=60)
    plt.title("Residuals (pred_log - actual_log)")
    plt.xlabel("Residual (log space)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def plot_pred_vs_actual_log(df: pd.DataFrame, outpath: Path) -> None:
    plt.figure()
    plt.scatter(df["target_log"], df["pred_log"], s=3, alpha=0.2)
    plt.title("Predicted vs Actual (log1p EUR)")
    plt.xlabel("Actual log1p(EUR)")
    plt.ylabel("Predicted log1p(EUR)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def plot_calibration_deciles(cal: pd.DataFrame, outpath: Path) -> None:
    plt.figure()
    plt.plot(cal["pred_mean"], cal["actual_mean"], marker="o")
    plt.title("Calibration (deciles): mean predicted EUR vs mean actual EUR")
    plt.xlabel("Mean predicted EUR")
    plt.ylabel("Mean actual EUR")
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def main() -> None:
    ensure_dirs()

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}. Run scripts/03_train_market_value.py first.")

    df = load_test_frame()

    required = ["date", "target_log", "market_value_in_eur"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in DuckDB table: {missing}")

    drop_cols = {
        "market_value_in_eur",
        "target_log",
        "date",
        "player_id",
        "name",
        "full_name",
    }
    feature_cols = [c for c in df.columns if c not in drop_cols]

    df = df.dropna(subset=["target_log"]).copy()

    X = df[feature_cols].copy()
    X = X.fillna(0)

    model = load(MODEL_PATH)

    df["pred_log"] = model.predict(X)
    df["resid_log"] = df["pred_log"] - df["target_log"]

    # pandas uses clip(lower=...), not clip(min=...)
    df["pred_eur"] = pd.Series(np.expm1(df["pred_log"].to_numpy())).clip(lower=0)
    df["actual_eur"] = df["market_value_in_eur"].clip(lower=0)

    df["abs_err_eur"] = (df["pred_eur"] - df["actual_eur"]).abs()
    df["signed_err_eur"] = df["pred_eur"] - df["actual_eur"]

    if "position" in df.columns:
        df["position_bucket"] = df["position"].apply(make_position_bucket)
    else:
        df["position_bucket"] = "Unknown"

    if "domestic_competition_id" not in df.columns:
        df["domestic_competition_id"] = "Unknown"

    by_pos = (
        df.groupby("position_bucket", dropna=False)
        .agg(
            n=("actual_eur", "count"),
            mae_eur=("abs_err_eur", "mean"),
            median_abs_err=("abs_err_eur", "median"),
            mean_signed_err=("signed_err_eur", "mean"),
        )
        .sort_values("mae_eur", ascending=False)
        .reset_index()
    )
    by_pos.to_csv(ARTIFACTS_DIR / f"error_by_position_{MODEL_VERSION}.csv", index=False)

    by_league = (
        df.groupby("domestic_competition_id", dropna=False)
        .agg(
            n=("actual_eur", "count"),
            mae_eur=("abs_err_eur", "mean"),
            median_abs_err=("abs_err_eur", "median"),
            mean_signed_err=("signed_err_eur", "mean"),
        )
        .sort_values("mae_eur", ascending=False)
        .reset_index()
    )
    by_league.to_csv(ARTIFACTS_DIR / f"error_by_league_{MODEL_VERSION}.csv", index=False)

    cols_keep = [
        "date",
        "player_id" if "player_id" in df.columns else None,
        "position" if "position" in df.columns else None,
        "domestic_competition_id",
        "actual_eur",
        "pred_eur",
        "signed_err_eur",
        "abs_err_eur",
    ]
    cols_keep = [c for c in cols_keep if c is not None]

    top_over = df.sort_values("signed_err_eur", ascending=False).head(50)[cols_keep]
    top_under = df.sort_values("signed_err_eur", ascending=True).head(50)[cols_keep]

    top_over.to_csv(ARTIFACTS_DIR / f"top_overpredictions_{MODEL_VERSION}.csv", index=False)
    top_under.to_csv(ARTIFACTS_DIR / f"top_underpredictions_{MODEL_VERSION}.csv", index=False)

    plot_residual_hist(df, VISUALS_DIR / f"residual_hist_{MODEL_VERSION}.png")
    plot_pred_vs_actual_log(df, VISUALS_DIR / f"pred_vs_actual_log_{MODEL_VERSION}.png")

    df_cal = df.sort_values("pred_eur").copy()
    df_cal["decile"] = pd.qcut(df_cal["pred_eur"], 10, labels=False, duplicates="drop")

    cal = (
        df_cal.groupby("decile")
        .agg(
            pred_mean=("pred_eur", "mean"),
            actual_mean=("actual_eur", "mean"),
            n=("actual_eur", "count"),
        )
        .reset_index()
    )
    cal.to_csv(ARTIFACTS_DIR / f"calibration_deciles_{MODEL_VERSION}.csv", index=False)
    plot_calibration_deciles(cal, VISUALS_DIR / f"calibration_deciles_{MODEL_VERSION}.png")

    summary = {
        "n_test_rows": int(len(df)),
        "cutoff_date": CUTOFF_DATE,
        "mae_eur": float(df["abs_err_eur"].mean()),
        "median_abs_err_eur": float(df["abs_err_eur"].median()),
        "mean_signed_err_eur": float(df["signed_err_eur"].mean()),
    }
    with open(ARTIFACTS_DIR / f"error_analysis_summary_{MODEL_VERSION}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("✅ Error analysis complete.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
