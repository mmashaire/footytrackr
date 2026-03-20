from __future__ import annotations

from pathlib import Path
from typing import Union
import argparse
import logging

import numpy as np
import pandas as pd


# Default paths (can be overridden via CLI)
DEFAULT_IN_PATH = Path("data/features/player_value_features_v1.csv")
DEFAULT_OUT_PATH = Path("data/features/player_value_features_v2.csv")

Number = Union[int, float, np.number]


def setup_logging(level: str = "INFO") -> None:
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def safe_divide(numer: pd.Series, denom: Union[pd.Series, Number]) -> pd.Series:
    """
    Elementwise division that returns 0 when denom is 0 / missing.
    Supports denom as either a Series or a scalar.
    """
    if np.isscalar(denom):
        if float(denom) == 0.0:
            return pd.Series(np.zeros(len(numer)), index=numer.index, dtype=float)
        out = (numer / float(denom)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return out.astype(float)

    denom_safe = denom.replace(0, np.nan)
    out = (numer / denom_safe).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out.astype(float)


def add_rate_features(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """
    Adds rate features for a given window prefix (e.g. w180, w365).
    Expects these total columns to exist:
      - {prefix}_games_played
      - {prefix}_minutes_played
      - {prefix}_goals
      - {prefix}_assists
      - {prefix}_yellow_cards
      - {prefix}_red_cards
    """
    g = f"{prefix}_games_played"
    m = f"{prefix}_minutes_played"
    goals = f"{prefix}_goals"
    assists = f"{prefix}_assists"
    yc = f"{prefix}_yellow_cards"
    rc = f"{prefix}_red_cards"

    required = [g, m, goals, assists, yc, rc]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns for {prefix}: {missing}")

    # Per-game: simple conditioning on how often they actually appeared
    df[f"{prefix}_minutes_per_game"] = safe_divide(df[m], df[g])

    # Per-90: normalize by minutes played in the window
    # "90s played" = minutes / 90
    nineties_played = safe_divide(df[m], 90.0)

    df[f"{prefix}_goals_per90"] = safe_divide(df[goals], nineties_played)
    df[f"{prefix}_assists_per90"] = safe_divide(df[assists], nineties_played)
    df[f"{prefix}_g_plus_a_per90"] = safe_divide(df[goals] + df[assists], nineties_played)
    df[f"{prefix}_yellow_per90"] = safe_divide(df[yc], nineties_played)
    df[f"{prefix}_red_per90"] = safe_divide(df[rc], nineties_played)

    # Tiny but often helpful: did the player play at all in the window?
    df[f"{prefix}_played_any_minutes"] = (df[m] > 0).astype(int)

    return df


def main(in_path: Path, out_path: Path) -> None:
    logging.info(f"Starting feature building from {in_path} to {out_path}")

    if not in_path.exists():
        raise FileNotFoundError(f"Missing input file: {in_path.resolve()}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path, parse_dates=["date"])
    logging.info(f"Loaded {len(df):,} rows from input file")

    # Add rate features for both windows
    df = add_rate_features(df, "w180")
    df = add_rate_features(df, "w365")
    logging.info("Added rate features for w180 and w365 windows")

    df.to_csv(out_path, index=False)
    logging.info(f"Saved v2 features to {out_path}")

    new_cols = [c for c in df.columns if ("per90" in c) or ("minutes_per_game" in c) or ("played_any_minutes" in c)]
    logging.info(f"Added {len(new_cols)} derived columns. Sample: {new_cols[:10]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build v2 player value features with rate calculations.")
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_IN_PATH,
        help="Path to input features CSV (default: data/features/player_value_features_v1.csv)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUT_PATH,
        help="Path to output features CSV (default: data/features/player_value_features_v2.csv)",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set logging level (default: INFO)",
    )

    args = parser.parse_args()
    setup_logging(args.log_level)
    main(args.input, args.output)
