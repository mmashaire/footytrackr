from __future__ import annotations

from pathlib import Path
import pandas as pd


IN_PATH = Path("data/features/player_value_features_v2.csv")
OUT_PATH = Path("data/features/player_value_features_v3.csv")


def collapse_rare_categories(
    df: pd.DataFrame,
    col: str,
    top_k: int = 30,
    other_label: str = "Other",
) -> pd.DataFrame:
    """
    Keep only the top_k most frequent categories; everything else becomes "Other".
    Also normalizes missing values to "Unknown" first.
    """
    if col not in df.columns:
        raise KeyError(f"Missing column: {col}")

    s = df[col].fillna("Unknown").astype(str).str.strip()
    top_values = s.value_counts(dropna=False).head(top_k).index

    df[col] = s.where(s.isin(top_values), other_label)
    return df


def main() -> None:
    if not IN_PATH.exists():
        raise FileNotFoundError(f"Missing input file: {IN_PATH.resolve()}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(IN_PATH, parse_dates=["date"])

    # High-cardinality categoricals: collapse rare values
    df = collapse_rare_categories(df, "country_of_birth", top_k=40)
    df = collapse_rare_categories(df, "country_of_citizenship", top_k=40)

    # League id also has many categories; keep a bit more because it matters
    df = collapse_rare_categories(df, "player_club_domestic_competition_id", top_k=60)

    df.to_csv(OUT_PATH, index=False)

    print(f"Saved v3 features: {OUT_PATH.as_posix()}")
    print(f"Rows: {len(df):,} | Columns: {len(df.columns):,}")
    print("Top country_of_birth categories after grouping:")
    print(df["country_of_birth"].value_counts().head(15).to_string())


if __name__ == "__main__":
    main()
