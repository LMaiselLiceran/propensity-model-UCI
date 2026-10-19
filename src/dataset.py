"""Data helpers for the bank propensity project."""

import sqlite3
from pathlib import Path
import pandas as pd

# Resolve the database path from this file's location
REPO_ROOT = Path(__file__).resolve().parents[1]
DB_PATH   = REPO_ROOT / 'data' / 'bank.db'

# The file contains no January or February contacts.
MONTHS = {'mar': 3, 'apr': 4, 'may': 5, 'jun': 6, 'jul': 7,
          'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}

def load_clients(db_path: Path = DB_PATH) -> pd.DataFrame:
    """Return the clients table in original file order.
    Remove duplicate rows in the process."""
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql("SELECT * FROM clients ORDER BY row_id", conn)
    finally:
        conn.close()
    cols = df.columns.drop("row_id")
    return df.drop_duplicates(subset=cols, keep="first").reset_index(drop=True)

def add_year_block(df: pd.DataFrame) -> pd.DataFrame:
    """Add 'year_block' (0, 1, 2, ...):
    a new block starts whenever the month number decreases"""
    month_num = df['month'].map(MONTHS)
    out = df.copy()
    out['year_block'] = (month_num < month_num.shift()).cumsum()
    return out

def time_split(df: pd.DataFrame, test_frac: float = 0.2):
    """Split by row order: first rows train, last 'test_frac' of rows test."""
    assert df['row_id'].is_monotonic_increasing, 'df must be sorted by row_id'
    cut = int(len(df) * (1 - test_frac))
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()