from pathlib import Path
import duckdb

FEATURE_VERSION = "v3"

ROOT = Path(__file__).resolve().parents[1]
FEATURES_PATH = ROOT / "data" / "features" / f"player_value_features_{FEATURE_VERSION}.csv"
DB_DIR = ROOT / "data" / "db"
DB_PATH = DB_DIR / "footytrackr.duckdb"

def main():
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Missing features file: {FEATURES_PATH}")

    DB_DIR.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(str(DB_PATH))

    # Drop + recreate so it’s deterministic
    con.execute("DROP TABLE IF EXISTS player_value_features;")

    # DuckDB reads CSV fast and infers types well enough for analytics
    con.execute(f"""
        CREATE TABLE player_value_features AS
        SELECT *
        FROM read_csv_auto('{FEATURES_PATH.as_posix()}',
            header=true,
            sample_size=200000,
            ignore_errors=true
        );
    """)

    # Helpful indexes (DuckDB supports them; mileage varies but ok)
    # Only do if these columns exist in your feature table:
    for col in ["date", "player_id", "position_group", "domestic_competition_id"]:
        try:
            con.execute(f"CREATE INDEX IF NOT EXISTS idx_{col} ON player_value_features({col});")
        except Exception:
            pass

    # Create a view that filters obviously bad rows (if needed)
    con.execute("DROP VIEW IF EXISTS v_model_rows;")
    con.execute("""
        CREATE VIEW v_model_rows AS
        SELECT *
        FROM player_value_features
        WHERE market_value_in_eur IS NOT NULL
          AND market_value_in_eur >= 0;
    """)

    row_count = con.execute("SELECT COUNT(*) FROM v_model_rows;").fetchone()[0]
    print(f"✅ Built DuckDB at: {DB_PATH}")
    print(f"✅ Rows in v_model_rows: {row_count:,}")

    con.close()

if __name__ == "__main__":
    main()
