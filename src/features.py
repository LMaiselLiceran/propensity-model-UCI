"""Shared feature definitions"""

import pandas as pd

TARGET = "subscribed"

CLIENT = ["age", "job", "marital", "education", "default", "housing", "loan"]
CONTACT = ["contact", "month", "day_of_week", "campaign", "pdays", "previous", "poutcome"]
MACRO = ["emp_var_rate", "cons_price_idx", "cons_conf_idx", "euribor3m", "nr_employed"]

# Left out on purpose:
#   duration: only known after the call ends (leakage)
#   row_id, year_block, month_num: information about time, not the client
FEATURES = CLIENT + CONTACT + MACRO

CATEGORICAL = ["job", "marital", "education", "default", "housing", "loan",
               "contact", "month", "day_of_week", "poutcome"]

def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Select the feature columns and encode pdays (999 = never contacted before)."""
    X = df[FEATURES].copy()
    X["contacted_before"] = (X["pdays"] != 999).astype(int)
    X["pdays"] = X["pdays"].where(X["pdays"] != 999, 0)
    return X

def numeric_columns(include_macro: bool = True) -> list:
    """Numeric feature columns, optionally without the macroeconomic indicators."""
    cols = [c for c in FEATURES if c not in CATEGORICAL]
    if not include_macro:
        cols = [c for c in cols if c not in MACRO]
    return cols + ["contacted_before"]