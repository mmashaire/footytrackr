from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
FEATURES_PATH = Path("data/features/player_value_features_v3.csv")
ARTIFACTS_DIR = Path("artifacts")

TARGET_COL = "target_log"
DATE_COL = "date"


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def artifact_tag_from_features_path(path: Path) -> str:
    """
    Derive a short tag like 'v1', 'v2', 'v3' from the features filename.
    Falls back to 'run' if pattern not found.
    """
    name = path.stem  # e.g. "player_value_features_v3"
    for tag in ("v1", "v2", "v3", "v4"):
        if name.endswith(tag):
            return tag
    return "run"


def time_based_split(df: pd.DataFrame, date_col: str, train_frac: float = 0.8):
    """
    Strict time split: train < cutoff, test >= cutoff.
    This avoids boundary overlap when many samples share the same date.
    """
    df = df.sort_values(date_col).reset_index(drop=True)
    split_idx = int(len(df) * train_frac)
    cutoff = df.loc[split_idx, date_col]

    train = df[df[date_col] < cutoff].copy()
    test = df[df[date_col] >= cutoff].copy()
    return train, test, cutoff


def evaluate(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    """
    Evaluate in log-space (model target) + show euro MAE for human readability.
    """
    rmse_log = root_mean_squared_error(y_true, y_pred)
    mae_log = mean_absolute_error(y_true, y_pred)

    y_true_eur = np.expm1(y_true)
    y_pred_eur = np.expm1(y_pred)
    mae_eur = mean_absolute_error(y_true_eur, y_pred_eur)

    return {
        "rmse_log": float(rmse_log),
        "mae_log": float(mae_log),
        "mae_eur": float(mae_eur),
    }


def drop_feature_groups(df: pd.DataFrame, mode: str | None) -> pd.DataFrame:
    """
    Remove groups of features for ablation experiments.

    Modes:
      - None / "full": keep everything
      - "no_nationality": drop country_of_birth + country_of_citizenship
      - "no_context": drop nationality + domestic competition id
      - "performance_only": keep age, position, and w180_/w365_ features only
    """
    if mode is None or mode == "full":
        return df

    out = df.copy()

    if mode == "no_nationality":
        out = out.drop(
            columns=["country_of_birth", "country_of_citizenship"],
            errors="ignore",
        )

    elif mode == "no_context":
        out = out.drop(
            columns=[
                "country_of_birth",
                "country_of_citizenship",
                "player_club_domestic_competition_id",
            ],
            errors="ignore",
        )

    elif mode == "performance_only":
        keep_cols = [
            "age",
            "position",
            *[c for c in out.columns if c.startswith("w180_")],
            *[c for c in out.columns if c.startswith("w365_")],
        ]
        # preserve only columns that actually exist
        keep_cols = [c for c in dict.fromkeys(keep_cols) if c in out.columns]
        out = out[keep_cols]

    else:
        raise ValueError(f"Unknown ablation mode: {mode}")

    return out


def build_model(X_train: pd.DataFrame) -> Pipeline:
    """
    Create a Ridge regression pipeline with:
      - numeric: median impute + standardize
      - categorical: most-frequent impute + one-hot
    """
    numeric_features = X_train.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X_train.select_dtypes(include=["object"]).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )

    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("ridge", Ridge(alpha=1.0)),
        ]
    )


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """
    Extract output feature names from a fitted ColumnTransformer in our setup:
      - numeric features pass through
      - categorical features expand via OneHotEncoder
    """
    names: list[str] = []

    # numeric block
    num_features = preprocessor.transformers_[0][2]
    names.extend(list(num_features))

    # categorical block
    cat_pipeline = preprocessor.transformers_[1][1]
    onehot = cat_pipeline.named_steps["onehot"]
    cat_features = preprocessor.transformers_[1][2]
    names.extend(list(onehot.get_feature_names_out(cat_features)))

    return names


# ---------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------
def baseline_global_median(y_train: pd.Series, y_test: pd.Series) -> dict:
    """Baseline #1: always predict the training median (log-space)."""
    pred = np.full(len(y_test), y_train.median())
    return evaluate(y_test, pred)


def baseline_position_age(df_train: pd.DataFrame, df_test: pd.DataFrame) -> dict:
    """
    Baseline #2: median by (position, age_bin).
    Strong baseline for valuation tasks.
    """
    train = df_train.copy()
    test = df_test.copy()

    age_bins = [0, 18, 21, 24, 27, 30, 35, 100]
    train["age_bin"] = pd.cut(train["age"], bins=age_bins, right=False)
    test["age_bin"] = pd.cut(test["age"], bins=age_bins, right=False)

    medians = (
        train.groupby(["position", "age_bin"], observed=False)[TARGET_COL]
        .median()
        .reset_index()
    )

    test = test.merge(
        medians,
        on=["position", "age_bin"],
        how="left",
        suffixes=("", "_pred"),
    )

    fallback = train[TARGET_COL].median()
    preds = test[f"{TARGET_COL}_pred"].fillna(fallback)

    return evaluate(test[TARGET_COL], preds)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> None:
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Missing features file: {FEATURES_PATH.resolve()}")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    df = (
        pd.read_csv(FEATURES_PATH, parse_dates=[DATE_COL])
        .sort_values(DATE_COL)
        .reset_index(drop=True)
    )

    tag = artifact_tag_from_features_path(FEATURES_PATH)
    model_path = ARTIFACTS_DIR / f"ridge_model_{tag}.joblib"
    top_path = ARTIFACTS_DIR / f"ridge_top_coefficients_{tag}.csv"
    metrics_path = ARTIFACTS_DIR / f"metrics_{tag}.json"
    ablation_path = ARTIFACTS_DIR / f"ablation_{tag}.csv"

    # Prevent leakage: remove identifiers + raw euro value + date
    drop_cols = ["player_id", "market_value_in_eur", "date_of_birth", DATE_COL]
    X = df.drop(columns=drop_cols)
    y = df[TARGET_COL].astype(float)

    # Combined frame for splitting + baselines
    df_model = X.copy()
    df_model[TARGET_COL] = y
    df_model[DATE_COL] = df[DATE_COL]

    df_train, df_test, cutoff = time_based_split(df_model, DATE_COL)

    X_train_full = df_train.drop(columns=[TARGET_COL, DATE_COL])
    y_train = df_train[TARGET_COL]
    X_test_full = df_test.drop(columns=[TARGET_COL, DATE_COL])
    y_test = df_test[TARGET_COL]

    # ----------------------------
    # Ablation study (same split)
    # ----------------------------
    experiments = [
        ("full", "full"),
        ("no_nationality", "no_nationality"),
        ("no_context", "no_context"),
        ("performance_only", "performance_only"),
    ]

    ablation_rows: list[dict] = []
    preds_full: np.ndarray | None = None
    model_full: Pipeline | None = None

    for exp_name, mode in experiments:
        X_train = drop_feature_groups(X_train_full, mode)
        X_test = drop_feature_groups(X_test_full, mode)

        model = build_model(X_train)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        metrics = evaluate(y_test, preds)
        ablation_rows.append(
            {
                "experiment": exp_name,
                "n_features_input": int(X_train.shape[1]),
                **metrics,
            }
        )

        if exp_name == "full":
            preds_full = preds
            model_full = model

    ablation_df = pd.DataFrame(ablation_rows).sort_values("rmse_log")
    ablation_df.to_csv(ablation_path, index=False)

    # Save the full model + coefficients (keep artifacts clean)
    if model_full is None or preds_full is None:
        raise RuntimeError("Full model training failed unexpectedly.")

    joblib.dump(model_full, model_path)

    fitted_pre = model_full.named_steps["preprocess"]
    fitted_ridge = model_full.named_steps["ridge"]
    feature_names = get_feature_names(fitted_pre)

    coefs = pd.Series(fitted_ridge.coef_, index=feature_names)
    top_coefs = pd.concat(
        [
            coefs.sort_values(ascending=False).head(25),
            coefs.sort_values(ascending=True).head(25),
        ]
    ).to_frame("coef").reset_index().rename(columns={"index": "feature"})

    top_coefs.to_csv(top_path, index=False)

    # Baselines computed once (same split)
    results = {
        "features_file": FEATURES_PATH.as_posix(),
        "cutoff_date": str(pd.to_datetime(cutoff).date()),
        "baseline_global_median": baseline_global_median(y_train, y_test),
        "baseline_position_age": baseline_position_age(df_train, df_test),
        "ridge_model_full": evaluate(y_test, preds_full),
        "ablation_path": ablation_path.as_posix(),
        "n_train": int(len(X_train_full)),
        "n_test": int(len(X_test_full)),
        "date_train_min": str(df_train[DATE_COL].min().date()),
        "date_train_max": str(df_train[DATE_COL].max().date()),
        "date_test_min": str(df_test[DATE_COL].min().date()),
        "date_test_max": str(df_test[DATE_COL].max().date()),
        "model_path": model_path.as_posix(),
        "top_coefficients_path": top_path.as_posix(),
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("=== Evaluation results ===")
    print(json.dumps(results, indent=2))
    print("\n=== Ablation study (sorted by RMSE) ===")
    print(ablation_df.to_string(index=False))

    print(f"\nSaved metrics  → {metrics_path.as_posix()}")
    print(f"Saved ablation → {ablation_path.as_posix()}")
    print(f"Saved model    → {model_path.as_posix()}")
    print(f"Saved coefs    → {top_path.as_posix()}")


if __name__ == "__main__":
    main()
