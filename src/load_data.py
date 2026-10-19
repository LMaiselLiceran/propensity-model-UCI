import sqlite3
from pathlib import Path

import pandas as pd
RAW_CSV = Path("data/raw/bank-additional-full.csv")
DB_PATH = Path("data/bank.db")

def load() -> pd.DataFrame:
    df = pd.read_csv(RAW_CSV, sep=";")
    # Make column names SQL-friendly
    df.columns = [c.replace(".", "_") for c in df.columns]
    # Rows are ordered chronologically but there's no explicit date column
    # Therefore insert a row_id column in the chronological order
    df.insert(0, "row_id", range(len(df)))
    # Make a new column using the "y" column (containing "yes"/"no") and assigning to 1 or 0
    df["subscribed"] = (df["y"] == "yes").astype(int)
    # Return a copy of the table without the "y" column
    return df.drop(columns=["y"])

def main() -> None:
    # Load the data as a pandas DataFrame
    df = load()
    # Create desired directories
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Connect and create a "clients" table with the data in df
    with sqlite3.connect(DB_PATH) as conn:
        df.to_sql("clients", conn, if_exists="replace", index=False)
    conn.close()
    print(f"Wrote {len(df)} lines to {DB_PATH}")
    # Give the subscription rate (use the 0/1 column we created)
    print(f"Subscription rate: {df['subscribed'].mean():.2%}")

# Only load the data if the file is run directly
if __name__ == "__main__":
    main()