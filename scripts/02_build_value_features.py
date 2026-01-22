from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/features")

PLAYERS_FILE = RAW_DIR / "players.csv"
VALUATIONS_FILE = RAW_DIR / "player_valuations.csv"
APPEARANCES_FILE = RAW_DIR / "appearances.csv"

OUT_PATH = OUT_DIR / "player_value_features_v1.csv"


def _require_files(paths: Iterable[Path]) -> None:
    missing = [p for p in paths if not p.exists()]
    if missing:
        msg = "Missing required file(s):\n" + "\n".join(f" - {p.resolve()}" for p in missing)
        raise FileNotFoundError(msg)


def _safe_to_datetime(series: pd.Series) -> pd.Series:
    # Coerce invalid dates to NaT (we'll drop later where needed)
    return pd.to_datetime(series, errors="coerce")


def _clean_foot(value: object) -> str:
    if pd.isna(value):
        return "Unknown"
    v = str(value).strip().lower()
    if v in {"right", "r"}:
        return "Right"
    if v in {"left", "l"}:
        return "Left"
    if v in {"both", "b"}:
        return "Both"
    return "Other"


def _clean_height_cm(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    # Conservative range; keep it simple and avoid weird outliers
    s = s.where((s >= 140) & (s <= 220))
    return s


def _height_bucket(h: float) -> str:
    if pd.isna(h):
        return "Unknown"
    if h < 170:
        return "Short"
    if h <= 185:
        return "Average"
    return "Tall"


def build_player_base(players: pd.DataFrame) -> pd.DataFrame:
    """Select and clean player attributes that should be safe at any valuation date."""
    keep_cols = [
        "player_id",
        "date_of_birth",
        "position",
        "sub_position",
        "foot",
        "height_in_cm",
        "country_of_citizenship",
        "country_of_birth",
    ]
    df = players.loc[:, [c for c in keep_cols if c in players.columns]].copy()

    df["date_of_birth"] = _safe_to_datetime(df["date_of_birth"])
    df["foot"] = df["foot"].apply(_clean_foot)
    df["height_in_cm"] = _clean_height_cm(df["height_in_cm"])
    df["height_bucket"] = df["height_in_cm"].apply(_height_bucket)

    # Normalize text fields a bit to reduce accidental category explosion
    for col in ["position", "sub_position", "country_of_citizenship", "country_of_birth"]:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown").astype(str).str.strip()

    return df


def add_age_at_valuation(df: pd.DataFrame) -> pd.DataFrame:
    """Compute age at valuation date. Rows with missing DOB or date will be dropped."""
    out = df.copy()
    out = out.dropna(subset=["date", "date_of_birth"])

    # Age in years (approx, but consistent). Using 365.25 accounts for leap years.
    age_days = (out["date"] - out["date_of_birth"]).dt.days
    out["age"] = (age_days / 365.25).astype(float)

    # Drop obviously broken ages (data errors)
    out = out[(out["age"] >= 12) & (out["age"] <= 50)]
    return out


def aggregate_appearances(
    valuations: pd.DataFrame,
    appearances: pd.DataFrame,
    window_days: int,
    prefix: str,
) -> pd.DataFrame:
    """
    Build rolling aggregates of performance in a fixed lookback window before each valuation.
    This is time-aware by construction: only appearances strictly before valuation date are counted.
    """
    vals = valuations.loc[:, ["player_id", "date"]].copy()
    vals["val_id"] = np.arange(len(vals), dtype=np.int64)

    apps = appearances.copy()
    apps["date"] = _safe_to_datetime(apps["date"])
    apps = apps.dropna(subset=["player_id", "date"])

    # Keep only needed numeric fields (safe defaults)
    num_cols = ["minutes_played", "goals", "assists", "yellow_cards", "red_cards"]
    for col in num_cols:
        if col in apps.columns:
            apps[col] = pd.to_numeric(apps[col], errors="coerce").fillna(0)
        else:
            apps[col] = 0

    # Merge to compare dates per valuation row
    merged = vals.merge(apps, on="player_id", how="left", suffixes=("", "_app"))

    # Only count appearances before the valuation date, within the lookback window
    start = merged["date"] - pd.to_timedelta(window_days, unit="D")
    in_window = (merged["date_app"] < merged["date"]) & (merged["date_app"] >= start)

    merged = merged.loc[in_window, ["val_id"] + num_cols]

    agg = (
        merged.groupby("val_id", as_index=False)
        .agg(
            games_played=("minutes_played", "size"),
            minutes_played=("minutes_played", "sum"),
            goals=("goals", "sum"),
            assists=("assists", "sum"),
            yellow_cards=("yellow_cards", "sum"),
            red_cards=("red_cards", "sum"),
        )
    )

    # Prefix columns so we can join multiple windows
    rename_map = {c: f"{prefix}_{c}" for c in agg.columns if c != "val_id"}
    agg = agg.rename(columns=rename_map)

    # Ensure every valuation row exists (fill missing with zeros)
    full = vals.loc[:, ["val_id"]].merge(agg, on="val_id", how="left")
    for c in full.columns:
        if c != "val_id":
            full[c] = full[c].fillna(0)

    return full


def main() -> None:
    _require_files([PLAYERS_FILE, VALUATIONS_FILE, APPEARANCES_FILE])
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    players = pd.read_csv(PLAYERS_FILE)
    valuations = pd.read_csv(VALUATIONS_FILE)
    appearances = pd.read_csv(APPEARANCES_FILE)

    valuations["date"] = _safe_to_datetime(valuations["date"])
    valuations = valuations.dropna(subset=["player_id", "date", "market_value_in_eur"]).copy()

    # Remove nonsensical targets (Transfermarkt can contain 0s; keep them if you want,
    # but for first pass it's cleaner to focus on strictly positive values)
    valuations["market_value_in_eur"] = pd.to_numeric(valuations["market_value_in_eur"], errors="coerce")
    valuations = valuations.dropna(subset=["market_value_in_eur"])
    valuations = valuations[valuations["market_value_in_eur"] > 0].copy()

    player_base = build_player_base(players)

    df = valuations.merge(player_base, on="player_id", how="left")

    # Age is computed after the merge so we have DOB available
    df = add_age_at_valuation(df)

    # Create an id so we can safely join appearance aggregates per valuation row
    df = df.sort_values(["date", "player_id"]).reset_index(drop=True)
    df["val_id"] = np.arange(len(df), dtype=np.int64)

    # Build performance aggregates
    perf_180 = aggregate_appearances(df.rename(columns={"val_id": "val_id"}), appearances, 180, "w180")
    perf_365 = aggregate_appearances(df.rename(columns={"val_id": "val_id"}), appearances, 365, "w365")

    # Join back
    df = df.merge(perf_180, on="val_id", how="left").merge(perf_365, on="val_id", how="left")

    # Target transforms
    df["target_log"] = np.log1p(df["market_value_in_eur"].astype(float))

    # Light final cleanup
    df = df.drop(columns=["val_id"])  # internal only
    df = df.reset_index(drop=True)

    df.to_csv(OUT_PATH, index=False)

    print(f"Saved features: {OUT_PATH.as_posix()}")
    print(f"Rows: {len(df):,} | Columns: {len(df.columns):,}")
    print("Sample columns:", list(df.columns)[:25])


if __name__ == "__main__":
    main()
