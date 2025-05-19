import os
import pandas as pd

# Path to raw CSV files
RAW_DATA_DIR = os.path.join("data", "raw")

def profile_csv(filepath):
    try:
        df = pd.read_csv(filepath, low_memory=False)
        print(f"\nFile: {os.path.basename(filepath)}")
        print(f"Rows: {df.shape[0]:,}, Columns: {df.shape[1]:,}")
        print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

        nulls = df.isnull().sum()
        nulls = nulls[nulls > 0]
        if not nulls.empty:
            print("Missing values:")
            print(nulls.sort_values(ascending=False))
        else:
            print("No missing values.")
    except Exception as e:
        print(f"\nError loading {os.path.basename(filepath)}: {e}")

def main():
    print("=== Profiling files in /data/raw/ ===\n")
    for file in os.listdir(RAW_DATA_DIR):
        if file.endswith(".csv"):
            profile_csv(os.path.join(RAW_DATA_DIR, file))

if __name__ == "__main__":
    main()
