import pandas as pd
import os

# Define the file path to the raw player data
RAW_FILE = os.path.join("data", "raw", "players.csv")

# Load the CSV file and handle any read errors gracefully
def load_players():
    try:
        df = pd.read_csv(RAW_FILE, low_memory=False)
        return df
    except Exception as e:
        print(f"Failed to load players.csv: {e}")
        return None

# Drop columns with too much missing data (over 50% by default)
def clean_missing_columns(df, threshold=0.5):
    missing_ratio = df.isnull().mean()
    drop_cols = missing_ratio[missing_ratio > threshold].index.tolist()
    
    print(f"Dropping columns with more than {int(threshold * 100)}% missing data:")
    print(drop_cols)
    
    return df.drop(columns=drop_cols)

# Build a full name column from first and last name, or fall back to raw 'name'
def build_full_name(row):
    if pd.notnull(row["first_name"]) and pd.notnull(row["last_name"]):
        return f"{row['first_name']} {row['last_name']}"
    elif pd.notnull(row["name"]):
        return row["name"]
    else:
        return None

# Standardize the 'foot' column values (e.g., left, right, both)
def normalize_foot(value):
    if isinstance(value, str):
        value = value.strip().lower()
        if "left" in value:
            return "Left"
        elif "right" in value:
            return "Right"
        elif "both" in value or "ambi" in value:
            return "Both"
    return None

def main():
    df = load_players()
    if df is None:
        return

    # Step 1: Drop columns with too much missing data
    print(f"Original shape: {df.shape}")
    df = clean_missing_columns(df)
    print(f"After dropping columns: {df.shape}")

    # Step 2: Create a 'full_name' column for readability
    df["full_name"] = df.apply(build_full_name, axis=1)
    print("Added full_name column.")

    # Step 3: Normalize the 'foot' column
    df["foot"] = df["foot"].apply(normalize_foot)
    print("Normalized 'foot' column.")

    # Step 4: Parse and clean 'date_of_birth'
    df["date_of_birth"] = pd.to_datetime(df["date_of_birth"], errors="coerce")
    before = len(df)
    df = df[df["date_of_birth"].notna()]
    after = len(df)
    print(f"Dropped {before - after} rows with invalid or missing date_of_birth.")

    # Step 5: Calculate age based on date_of_birth
    from datetime import datetime
    today = pd.Timestamp(datetime.today().date())
    df["age"] = (today - df["date_of_birth"]).dt.days // 365
    print("Calculated 'age' column.")

    # Step 6: Clean and filter 'height_in_cm'
    df["height_in_cm"] = pd.to_numeric(df["height_in_cm"], errors="coerce")
    before = len(df)
    df = df[df["height_in_cm"].between(150, 210, inclusive="both")]
    after = len(df)
    print(f"Dropped {before - after} rows with unrealistic height.")

    # Step 7: Add a new column to categorize height ranges
    def height_category(h):
        if h < 170:
            return "Short"
        elif h <= 185:
            return "Average"
        else:
            return "Tall"

    df["height_category"] = df["height_in_cm"].apply(height_category)
    print("Created height_category column.")

    # Step 8: Save the cleaned dataset to the processed folder
    output_path = os.path.join("data", "processed", "players_clean.csv")
    df.to_csv(output_path, index=False)
    print(f"Saved cleaned data to {output_path}")

    # Placeholder: Add more field cleanups later if needed

if __name__ == "__main__":
    main()
