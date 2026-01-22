from pathlib import Path
import pandas as pd

RAW_DIR = Path("data/raw")

def quick_profile_csv(path: Path, nrows: int = 5_000) -> dict:
    df = pd.read_csv(path, nrows=nrows)
    return {
        "file": path.name,
        "rows_sampled": len(df),
        "cols": len(df.columns),
        "columns": list(df.columns),
        "null_pct_top10": (
            df.isna().mean()
            .sort_values(ascending=False)
            .head(10)
            .round(3)
            .to_dict()
        ),
    }

def main():
    if not RAW_DIR.exists():
        raise SystemExit(f"Missing folder: {RAW_DIR.resolve()}")

    csvs = sorted(RAW_DIR.glob("*.csv"))
    if not csvs:
        raise SystemExit(f"No CSVs found in: {RAW_DIR.resolve()}")

    print(f"Found {len(csvs)} CSV files in {RAW_DIR}:\n")
    for p in csvs:
        print(" -", p.name)

    print("\n--- SAMPLE PROFILES ---\n")
    for p in csvs:
        try:
            info = quick_profile_csv(p)
            print(f"## {info['file']}")
            print(f"Sampled rows: {info['rows_sampled']}, cols: {info['cols']}")
            print("Columns:", info["columns"])
            print("Top null %:", info["null_pct_top10"])
            print()
        except Exception as e:
            print(f"## {p.name} FAILED: {e}\n")

if __name__ == "__main__":
    main()
