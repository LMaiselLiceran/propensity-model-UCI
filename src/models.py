"""Shared pipeline construction"""

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features import CATEGORICAL, numeric_columns

def build_pipeline(model, include_macro: bool = True, include_month: bool = True) -> Pipeline:
    """Preprocessing (one-hot enconding for categories and scaling for numbers)"""
    categorical = [c for c in CATEGORICAL if include_month or c != "month"]
    prep = ColumnTransformer(
        [
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical),
            ('num', StandardScaler(), numeric_columns(include_macro))
        ],
        sparse_threshold=0
    )
    return Pipeline([('prep', prep), ('model', model)])