import pandas as pd
import os

# Define the path to the raw club_games.csv file
RAW_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw", "club_games.csv")

def load_club_games():
    """Loads the raw club_games CSV file into a DataFrame."""
    try:
        df = pd.read_csv(RAW_FILE, low_memory=False)
        return df
    except Exception as e:
        print(f"Failed to load club_games.csv: {e}")
        return None

def clean_club_games(df):
    """Performs basic cleaning: drop broken rows and fill missing goal data."""
    print(f"Original shape: {df.shape}")
    
    # Drop rows where either club_id or opponent_id is missing
    before = len(df)
    df = df.dropna(subset=["club_id", "opponent_id"])
    after = len(df)
    print(f"Dropped {before - after} rows with missing club or opponent ID.")

    # Fill missing goal values with 0, then convert to integers
    df.loc[:, "own_goals"] = df["own_goals"].fillna(0).astype(int)
    df.loc[:, "opponent_goals"] = df["opponent_goals"].fillna(0).astype(int)
    print("Filled missing goal values with 0.")

    return df

def main():
    df = load_club_games()
    if df is None:
        return

    df = clean_club_games(df)

    # Save cleaned DataFrame to processed folder
    output_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed", "club_games_clean.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)  # Make sure the directory exists
    df.to_csv(output_path, index=False)
    print(f"Saved cleaned data to {output_path}")

if __name__ == "__main__":
    main()
