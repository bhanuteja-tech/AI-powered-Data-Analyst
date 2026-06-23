import os
from typing import Any, Dict

import pandas as pd

from backend.pipelines.legacy.data_cleaning import clean_dataset


def create_model_ready_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Builds a model-ready dataset on top of the professional cleaning pipeline.

    Steps:
    - Run full cleaning (clean_dataset)
    - Drop ID-like and low-information columns
    - Apply one-hot encoding to recommended categorical columns
    - Scale numeric features (z-score) where appropriate
    """
    cleaning_result = clean_dataset(df)
    cleaned_df: pd.DataFrame = cleaning_result["cleaned_df"].copy()
    steps = cleaning_result["detailed_steps"]

    id_like_cols = steps.get("id_like_columns", []) or []
    low_info_cols = steps.get("low_information_columns", []) or []
    encoding_candidates = steps.get("encoding_candidates", []) or []
    scaling_candidates = steps.get("scaling_candidates", []) or []

    # 1. Drop ID-like and low-information columns
    cols_to_drop = [c for c in set(id_like_cols + low_info_cols) if c in cleaned_df.columns]
    cleaned_df = cleaned_df.drop(columns=cols_to_drop, errors="ignore")

    # 2. One-hot encode categorical columns with reasonable cardinality
    categorical_for_encoding = [c for c in encoding_candidates if c in cleaned_df.columns]
    df_model = pd.get_dummies(cleaned_df, columns=categorical_for_encoding, drop_first=False)

    # 3. Scale numeric features with simple z-score (mean/std)
    numeric_for_scaling = [c for c in scaling_candidates if c in df_model.columns]
    scaling_info: Dict[str, Dict[str, float]] = {}
    for col in numeric_for_scaling:
        series = df_model[col].astype(float)
        mean = float(series.mean())
        std = float(series.std()) if series.std() not in (0, None) else 0.0
        if std > 0:
            df_model[col] = (series - mean) / std
            scaling_info[col] = {"mean": mean, "std": std}

    metadata: Dict[str, Any] = {
        "cleaning_summary": cleaning_result["cleaning_summary"],
        "dropped_columns": cols_to_drop,
        "one_hot_encoded_columns": categorical_for_encoding,
        "scaled_numeric_columns": scaling_info,
        "final_feature_count": df_model.shape[1],
        "final_row_count": df_model.shape[0],
    }

    return {
        "model_ready_df": df_model,
        "metadata": metadata,
    }

