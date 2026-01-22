from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd


IN_PATH = Path("data/features/player_value_features_v1.csv")
OUT_PATH = Path("data/features/player_value_features_v2.csv")


Number = Union[int, float, np.number]


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


def main() -> None:
    if not IN_PATH.exists():
        raise FileNotFoundError(f"Missing input file: {IN_PATH.resolve()}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(IN_PATH, parse_dates=["date"])

    # Add rate features for both windows
    df = add_rate_features(df, "w180")
    df = add_rate_features(df, "w365")

    df.to_csv(OUT_PATH, index=False)

    new_cols = [c for c in df.columns if ("per90" in c) or ("minutes_per_game" in c) or ("played_any_minutes" in c)]
    print(f"Saved v2 features: {OUT_PATH.as_posix()}")
    print(f"Rows: {len(df):,} | Columns: {len(df.columns):,}")
    print(f"Added {len(new_cols)} derived columns. Sample:")
    print(new_cols[:25])


if __name__ == "__main__":
    main()
