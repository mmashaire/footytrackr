import pandas as pd
import os

RAW_FILE = os.path.join("data", "raw", "players.csv")

def load_players():
    try:
        df = pd.read_csv(RAW_FILE, low_memory=False)
        return df
    except Exception as e:
        print(f"Failed to load players.csv: {e}")
        return None

def clean_missing_columns(df, threshold=0.5):
    missing_ratio = df.isnull().mean()
    drop_cols = missing_ratio[missing_ratio > threshold].index.tolist()
    
    print(f"Dropping columns with more than {int(threshold * 100)}% missing data:")
    print(drop_cols)
    
    return df.drop(columns=drop_cols)

def build_full_name(row):
    if pd.notnull(row["first_name"]) and pd.notnull(row["last_name"]):
        return f"{row['first_name']} {row['last_name']}"
    elif pd.notnull(row["name"]):
        return row["name"]
    else:
        return None

def main():
    df = load_players()
    if df is None:
        return

    print(f"Original shape: {df.shape}")
    df = clean_missing_columns(df)
    print(f"After dropping columns: {df.shape}")

    df["full_name"] = df.apply(build_full_name, axis=1)
    print("Added full_name column.")

    # Placeholder for more cleaning steps later
    # (e.g., standardizing nationalities, normalizing height, etc.)

if __name__ == "__main__":
    main()
