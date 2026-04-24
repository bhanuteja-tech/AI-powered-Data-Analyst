import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple


def _identify_column_types(df: pd.DataFrame) -> Dict[str, List[str]]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    # Avoid select_dtypes strings for tz-aware dtypes; use API check (same as preprocess.py fix).
    datetime_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    return {
        "numeric": numeric_cols,
        "categorical": categorical_cols,
        "datetime": datetime_cols,
    }


def _convert_data_types(df: pd.DataFrame, report: Dict[str, Any]) -> pd.DataFrame:
    dtype_changes: Dict[str, str] = {}

    # Attempt to convert obvious numeric-like object columns
    for col in df.columns:
        if df[col].dtype == "object":
            sample = df[col].dropna().astype(str).head(20)
            if sample.empty:
                continue

            # Heuristic: if most non-null values look numeric, coerce to numeric
            numeric_like = sample.str.fullmatch(r"[-+]?\d*\.?\d+").mean()
            if numeric_like >= 0.8:
                before = str(df[col].dtype)
                df[col] = pd.to_numeric(df[col], errors="coerce")
                after = str(df[col].dtype)
                if before != after:
                    dtype_changes[col] = f"{before} -> {after}"

    # Attempt to convert date-like columns
    for col in df.columns:
        if df[col].dtype == "object":
            lower_name = col.lower()
            looks_like_date_name = any(
                kw in lower_name for kw in ["date", "time", "year", "month", "day"]
            )
            if not looks_like_date_name:
                continue

            before = str(df[col].dtype)
            converted = pd.to_datetime(df[col], errors="coerce", infer_datetime_format=True)
            # Only adopt conversion if we get a reasonable fraction of non-null datetimes
            if converted.notna().mean() >= 0.6:
                df[col] = converted
                after = str(df[col].dtype)
                if before != after:
                    dtype_changes[col] = f"{before} -> {after}"

    report["dtype_changes"] = dtype_changes
    return df


def _standardize_categoricals(df: pd.DataFrame, categorical_cols: List[str], report: Dict[str, Any]) -> pd.DataFrame:
    standardized_cols: List[str] = []
    for col in categorical_cols:
        if col not in df.columns:
            continue
        if not pd.api.types.is_string_dtype(df[col]) and not pd.api.types.is_categorical_dtype(df[col]):
            continue

        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .str.replace(r"\s+", " ", regex=True)
        )
        standardized_cols.append(col)

    report["standardized_categorical_columns"] = standardized_cols
    return df


def _handle_missing_values(df: pd.DataFrame, col_types: Dict[str, List[str]], report: Dict[str, Any]) -> Tuple[pd.DataFrame, List[str]]:
    missing_summary = df.isna().sum().to_dict()
    n_rows = len(df)
    drop_candidates: List[str] = []

    for col, count in missing_summary.items():
        if n_rows == 0:
            continue
        missing_pct = count / n_rows
        if missing_pct > 0.4:
            drop_candidates.append(col)
            continue

        if count == 0:
            continue

        if col in col_types["numeric"]:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
        elif col in col_types["categorical"]:
            mode_series = df[col].mode()
            if not mode_series.empty:
                df[col] = df[col].fillna(mode_series.iloc[0])

    if drop_candidates:
        df = df.drop(columns=drop_candidates)

    report["missing_values_per_column"] = missing_summary
    report["dropped_columns_high_missing"] = drop_candidates
    return df, drop_candidates


def _handle_duplicates(df: pd.DataFrame, report: Dict[str, Any]) -> pd.DataFrame:
    initial_rows = len(df)
    df = df.drop_duplicates()
    removed = initial_rows - len(df)
    report["duplicate_rows_removed"] = int(removed)
    return df


def _handle_outliers_iqr(df: pd.DataFrame, numeric_cols: List[str], report: Dict[str, Any]) -> pd.DataFrame:
    outlier_info: Dict[str, Dict[str, Any]] = {}

    for col in numeric_cols:
        series = df[col].dropna()
        if series.empty:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        # Cap extreme values rather than dropping rows
        before_extremes = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
        after_extremes = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()

        if before_extremes > 0:
            outlier_info[col] = {
                "method": "IQR capping",
                "lower_bound": float(lower_bound),
                "upper_bound": float(upper_bound),
                "values_capped": int(before_extremes - after_extremes),
            }

    report["outlier_treatment"] = outlier_info
    return df


def _feature_recommendations(
    df: pd.DataFrame,
    col_types: Dict[str, List[str]],
    dropped_cols_high_missing: List[str],
    report: Dict[str, Any],
) -> List[str]:
    low_info_cols: List[str] = []
    id_like_cols: List[str] = []
    encoding_candidates: List[str] = []
    scaling_candidates: List[str] = []

    for col in df.columns:
        unique_vals = df[col].nunique(dropna=False)

        # Heuristic: ID-like columns
        if unique_vals == len(df) and any(
            kw in col.lower() for kw in ["id", "uuid", "guid", "key"]
        ):
            id_like_cols.append(col)

        # Very low information columns (single value)
        if unique_vals <= 1:
            low_info_cols.append(col)

    for col in col_types["categorical"]:
        if col in df.columns:
            unique_vals = df[col].nunique(dropna=False)
            # Reasonable cardinality for one-hot encoding
            if 2 <= unique_vals <= 20:
                encoding_candidates.append(col)

    for col in col_types["numeric"]:
        if col in df.columns:
            scaling_candidates.append(col)

    report["id_like_columns"] = id_like_cols
    report["low_information_columns"] = low_info_cols
    report["encoding_candidates"] = encoding_candidates
    report["scaling_candidates"] = scaling_candidates

    # Recommended features for modeling: all remaining non-id numeric and categorical columns
    recommended = [
        col
        for col in df.columns
        if col not in id_like_cols and col not in low_info_cols and col not in dropped_cols_high_missing
    ]
    report["recommended_features"] = recommended
    return recommended


def clean_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """
    End-to-end professional data cleaning pipeline.
    Returns a report dict with:
      - cleaned_df (DataFrame)
      - cleaning_summary (str)
      - recommended_features (list[str])
      - detailed_steps (dict with per-step metadata)
    """
    original_shape = df.shape
    report: Dict[str, Any] = {"original_shape": original_shape}

    # 1. Dataset Overview
    report["columns"] = df.columns.tolist()
    report["dtypes"] = {col: str(dtype) for col, dtype in df.dtypes.items()}
    report["head"] = df.head(5).to_dict(orient="records")

    # 2. Data Profiling
    col_types = _identify_column_types(df)
    report["column_types"] = col_types

    # Be defensive: some datasets have no numeric or no describable columns
    summary_numeric: Dict[str, Any] = {}
    try:
        desc_num = df.describe()
        if not desc_num.empty:
            summary_numeric = desc_num.to_dict()
    except Exception:
        summary_numeric = {}
    report["summary_numeric"] = summary_numeric

    summary_all: Dict[str, Any] = {}
    try:
        with pd.option_context("display.max_colwidth", 200):
            # Avoid newer pandas-only kwargs to keep compatibility
            desc_all = df.describe(include="all")
            if not desc_all.empty:
                summary_all = desc_all.to_dict()
    except Exception:
        summary_all = {}
    report["summary_all"] = summary_all

    # 3. Missing Value Analysis (with 4. Duplicate Records integrated later)
    df, dropped_cols_high_missing = _handle_missing_values(df, col_types, report)

    # 4. Duplicate Records
    df = _handle_duplicates(df, report)

    # 5. Data Type Correction
    df = _convert_data_types(df, report)
    col_types = _identify_column_types(df)

    # 6. Outlier Detection & Treatment
    df = _handle_outliers_iqr(df, col_types["numeric"], report)

    # 7. Inconsistent Values (categorical standardization)
    df = _standardize_categoricals(df, col_types["categorical"], report)

    # 8–10. Feature Encoding / Scaling / Selection – recommendations only
    recommended_features = _feature_recommendations(
        df, col_types, dropped_cols_high_missing, report
    )

    # 11. Data Validation
    validation = {
        "no_missing_values": bool(df.isna().sum().sum() == 0),
        "row_count": int(df.shape[0]),
        "column_count": int(df.shape[1]),
        "dtypes_after": {col: str(dtype) for col, dtype in df.dtypes.items()},
    }
    report["validation"] = validation

    # Build human-readable summary
    steps_summary: List[str] = []
    steps_summary.append(
        f"Dataset started with {original_shape[0]} rows and {original_shape[1]} columns."
    )
    if report.get("missing_values_per_column"):
        steps_summary.append("Handled missing values per column using median (numeric) and mode (categorical).")
    if dropped_cols_high_missing:
        steps_summary.append(
            "Dropped columns with more than 40% missing values: "
            + ", ".join(dropped_cols_high_missing)
        )
    if report.get("duplicate_rows_removed", 0) > 0:
        steps_summary.append(
            f"Removed {report['duplicate_rows_removed']} duplicate rows based on full-row duplicates."
        )
    if report.get("dtype_changes"):
        steps_summary.append(
            "Corrected data types for columns: "
            + "; ".join(f"{c} ({chg})" for c, chg in report["dtype_changes"].items())
        )
    if report.get("outlier_treatment"):
        treated_cols = list(report["outlier_treatment"].keys())
        steps_summary.append(
            "Detected and capped outliers using IQR method for numeric columns: "
            + ", ".join(treated_cols)
        )
    if report.get("standardized_categorical_columns"):
        steps_summary.append(
            "Standardized categorical text values (trimmed whitespace, normalized spacing) for: "
            + ", ".join(report["standardized_categorical_columns"])
        )
    if report.get("encoding_candidates"):
        steps_summary.append(
            "Identified categorical columns suitable for one-hot encoding: "
            + ", ".join(report["encoding_candidates"])
        )
    if report.get("scaling_candidates"):
        steps_summary.append(
            "Identified numeric columns suitable for feature scaling: "
            + ", ".join(report["scaling_candidates"])
        )
    if report.get("id_like_columns") or report.get("low_information_columns"):
        drops = []
        if report.get("id_like_columns"):
            drops.append(
                "ID-like columns (" + ", ".join(report["id_like_columns"]) + ")"
            )
        if report.get("low_information_columns"):
            drops.append(
                "low-information columns (" + ", ".join(report["low_information_columns"]) + ")"
            )
        steps_summary.append(
            "Flagged columns that are likely not useful for modeling: " + "; ".join(drops)
        )

    steps_summary.append(
        f"Final cleaned dataset has {df.shape[0]} rows and {df.shape[1]} columns."
    )

    cleaning_summary = "\n".join(f"{i+1}. {line}" for i, line in enumerate(steps_summary))

    return {
        "cleaned_df": df,
        "cleaning_summary": cleaning_summary,
        "recommended_features": recommended_features,
        "detailed_steps": report,
    }

