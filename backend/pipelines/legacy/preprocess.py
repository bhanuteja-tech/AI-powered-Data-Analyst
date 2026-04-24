from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import re

import numpy as np
import pandas as pd


_TARGET_NAMES = {"target", "label", "output", "y"}
_ID_LIKE_PATTERNS = [
    re.compile(r"\bid\b", re.IGNORECASE),
    re.compile(r"user[_\s]*id", re.IGNORECASE),
    re.compile(r"customer[_\s]*id", re.IGNORECASE),
    re.compile(r"account[_\s]*id", re.IGNORECASE),
    re.compile(r"transaction[_\s]*id", re.IGNORECASE),
    re.compile(r"uuid", re.IGNORECASE),
    re.compile(r"guid", re.IGNORECASE),
    re.compile(r"key", re.IGNORECASE),
]


def _detect_target_column(df: pd.DataFrame) -> Optional[str]:
    # 1) explicit target names
    for col in df.columns:
        if col is None:
            continue
        if str(col).strip().lower() in _TARGET_NAMES:
            return col

    # 2) heuristic inference: single binary-ish column
    candidates: List[str] = []
    for col in df.columns:
        s = df[col]
        non_null = s.dropna()
        if non_null.empty:
            continue
        nunique = non_null.nunique(dropna=True)
        if nunique == 2:
            # binary classification-like
            candidates.append(col)

    if len(candidates) == 1:
        return candidates[0]

    # ambiguous or none -> proceed without target
    return None


def _standardize_categorical_text(df: pd.DataFrame, cols: List[str]) -> List[str]:
    standardized: List[str] = []
    for col in cols:
        if col not in df.columns:
            continue
        if not (pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col])):
            continue
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .str.replace(r"\s+", " ", regex=True)
            .str.lower()
        )
        standardized.append(col)
    return standardized


def _convert_numeric_like_columns(df: pd.DataFrame, report: Dict[str, Any]) -> None:
    # Convert object columns that look numeric to numeric dtype.
    # This keeps conversion safe via a high "numeric-like" fraction heuristic.
    numeric_like_regex = re.compile(r"^[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?$")
    dtype_changes: Dict[str, str] = {}

    for col in df.columns:
        if df[col].dtype != "object" and not pd.api.types.is_string_dtype(df[col]):
            continue
        sample = df[col].dropna().astype(str).head(50)
        if sample.empty:
            continue

        is_numeric = sample.map(lambda x: bool(numeric_like_regex.match(str(x).strip()))).mean()
        if is_numeric < 0.8:
            continue

        before = str(df[col].dtype)
        converted = pd.to_numeric(df[col], errors="coerce")
        after = str(converted.dtype)
        if before != after:
            df[col] = converted
            dtype_changes[col] = f"{before} -> {after}"

    report["dtype_changes_numeric_like"] = dtype_changes


def _convert_boolean_like_columns(df: pd.DataFrame, report: Dict[str, Any]) -> None:
    # Convert object columns with boolean-like tokens to actual bool dtype.
    mapping = {
        "true": True,
        "false": False,
        "yes": True,
        "no": False,
        "y": True,
        "n": False,
        "t": True,
        "f": False,
        "1": True,
        "0": False,
    }

    dtype_changes: Dict[str, str] = {}
    bool_like_columns: List[str] = []

    for col in df.columns:
        if df[col].dtype != "object" and not pd.api.types.is_string_dtype(df[col]):
            continue
        sample_series = df[col].dropna().astype(str).str.strip().str.lower()
        if sample_series.empty:
            continue

        unique_vals = set(sample_series.unique().tolist())
        known = set(mapping.keys())
        if not unique_vals.issubset(known):
            # if it's mostly boolean-like, we still convert
            ratio = sample_series.map(lambda x: x in known).mean()
            if ratio < 0.9:
                continue

        before = str(df[col].dtype)
        converted = df[col].astype(str).str.strip().str.lower().map(mapping)
        # Only keep conversion if it yields enough non-null values
        non_null_ratio = converted.notna().mean()
        if non_null_ratio < 0.9:
            continue

        df[col] = converted.astype("boolean")
        dtype_changes[col] = f"{before} -> boolean"
        bool_like_columns.append(col)

    report["dtype_changes_boolean_like"] = dtype_changes
    report["boolean_like_columns_converted"] = bool_like_columns


def _convert_datetime_columns(df: pd.DataFrame, report: Dict[str, Any]) -> None:
    # Convert likely date columns by name heuristics.
    dtype_changes: Dict[str, str] = {}
    converted_cols: List[str] = []

    for col in df.columns:
        if df[col].dtype != "object" and not pd.api.types.is_string_dtype(df[col]):
            continue
        lower_name = str(col).lower()
        if not any(kw in lower_name for kw in ["date", "time", "year", "month", "day"]):
            continue

        converted = pd.to_datetime(df[col], errors="coerce", infer_datetime_format=True)
        non_null_ratio = converted.notna().mean() if len(converted) else 0
        if non_null_ratio >= 0.6:
            before = str(df[col].dtype)
            df[col] = converted
            dtype_changes[col] = f"{before} -> datetime64[ns]"
            converted_cols.append(col)

    report["dtype_changes_datetime"] = dtype_changes
    report["datetime_columns_converted"] = converted_cols


def _drop_high_risk_columns(
    df: pd.DataFrame,
    target_col: Optional[str],
    report: Dict[str, Any],
    missing_threshold: float = 0.5,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    drop_cols: List[str] = []

    # Missingness
    n_rows = len(df)
    missing_pct = df.isna().sum() / n_rows if n_rows else pd.Series(dtype=float)
    high_missing = missing_pct[missing_pct > missing_threshold].index.tolist()

    # ID-like columns
    id_like: List[str] = []
    for col in df.columns:
        if col == target_col:
            continue
        col_str = str(col)
        if any(p.search(col_str) for p in _ID_LIKE_PATTERNS):
            id_like.append(col)

    # Constant columns
    constant_cols: List[str] = []
    for col in df.columns:
        if col == target_col:
            continue
        if df[col].nunique(dropna=False) <= 1:
            constant_cols.append(col)

    # Remove exact duplicate columns (same meaning)
    dup_cols: List[str] = []
    # Create lightweight signatures without mutating df
    sig_df = df.copy()
    for col in sig_df.columns:
        if pd.api.types.is_datetime64_any_dtype(sig_df[col]):
            # tz-aware columns: avoid .astype("int64") (can fail); string is stable for duplicate keys
            sig_df[col] = pd.to_datetime(sig_df[col], errors="coerce").astype(str)
    sig_df = sig_df.astype(str).fillna("__MISSING__")
    dup_mask = sig_df.T.duplicated(keep="first")
    dup_cols = sig_df.columns[dup_mask].tolist()
    dup_cols = [c for c in dup_cols if c != target_col]

    drop_cols = sorted(set(high_missing + id_like + constant_cols + dup_cols))

    report["dropped_columns_missing_gt"] = high_missing
    report["dropped_columns_id_like"] = id_like
    report["dropped_columns_constant"] = constant_cols
    report["dropped_columns_exact_duplicates"] = dup_cols
    report["dropped_columns_total"] = drop_cols

    return df.drop(columns=drop_cols, errors="ignore"), {"dropped_columns": drop_cols}


def _cap_outliers_iqr(df: pd.DataFrame, numeric_cols: List[str], report: Dict[str, Any]) -> None:
    outlier_treatment: Dict[str, Dict[str, Any]] = {}

    for col in numeric_cols:
        s = pd.to_numeric(df[col], errors="coerce")
        series = s.dropna()
        if series.empty:
            continue
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0 or pd.isna(iqr):
            continue

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        before = ((s < lower) | (s > upper)).sum()
        df[col] = s.clip(lower=lower, upper=upper)
        after = ((df[col] < lower) | (df[col] > upper)).sum()

        if before and before > after:
            outlier_treatment[col] = {
                "method": "IQR capping",
                "lower_bound": float(lower),
                "upper_bound": float(upper),
                "values_capped": int(before - after),
            }

    report["outlier_treatment_iqr"] = outlier_treatment


def _impute_missing_adaptive(df: pd.DataFrame, col_types: Dict[str, List[str]], report: Dict[str, Any]) -> List[str]:
    dropped_cols = report.get("dropped_columns_total", []) or []
    n_rows = len(df)
    handled_cols: List[str] = []

    for col in df.columns:
        if col in dropped_cols:
            continue
        if df[col].isna().sum() == 0:
            continue

        if col in col_types["numeric"]:
            non_null = df[col].dropna()
            if non_null.empty:
                fill_val = 0
            else:
                # adaptive: skew -> median else mean
                skew = non_null.skew()
                if abs(skew) > 0.5:
                    fill_val = float(non_null.median())
                else:
                    fill_val = float(non_null.mean())
            df[col] = df[col].fillna(fill_val)
            handled_cols.append(col)
        elif col in col_types["boolean"]:
            mode_series = df[col].mode(dropna=True)
            fill_val = bool(mode_series.iloc[0]) if not mode_series.empty else False
            df[col] = df[col].fillna(fill_val)
            handled_cols.append(col)
        elif col in col_types["categorical"]:
            mode_series = df[col].mode(dropna=True)
            fill_val = mode_series.iloc[0] if not mode_series.empty else "unknown"
            df[col] = df[col].fillna(fill_val)
            handled_cols.append(col)

    report["imputed_missing_columns"] = handled_cols
    return handled_cols


def _identify_column_types_for_preprocessing(df: pd.DataFrame) -> Dict[str, List[str]]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    boolean_cols = df.select_dtypes(include=["bool", "boolean"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    # Never use select_dtypes(include=["datetime64[ns, tz]"]): NumPy rejects metadata "[ns, tz]".
    datetime_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    return {
        "numeric": numeric_cols,
        "boolean": boolean_cols,
        "categorical": categorical_cols,
        "datetime": datetime_cols,
    }


def preprocess_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Returns BOTH:
      - cleaned_df (no encoding/scaling)
      - model_ready_df (encoded/scaled)

    Follows your spec for adaptive, generalized, safe preprocessing.
    """
    report: Dict[str, Any] = {}
    original_shape = df.shape

    # Defensive copy
    df_work = df.copy()

    # STEP 1: Understand dataset
    # ===== TARGET SEPARATION (SAFE) =====
    target_col = _detect_target_column(df_work)

    if target_col and target_col in df_work.columns:
        y = df_work[target_col].copy()
        X = df_work.drop(columns=[target_col]).copy()
    else:
        y = None
        X = df_work.copy()

    col_types_before = _identify_column_types_for_preprocessing(df_work)
    report["column_types_before"] = col_types_before
    report["original_shape"] = original_shape

    # Data type correction (numeric-like, boolean-like, datetime-like)
    _convert_numeric_like_columns(df_work, report)
    _convert_boolean_like_columns(df_work, report)
    _convert_datetime_columns(df_work, report)

    # Standardize categorical text (consistent labels)
    categorical_cols = df_work.select_dtypes(include=["object", "category"]).columns.tolist()
    standardized = _standardize_categorical_text(df_work, categorical_cols)
    report["standardized_categorical_columns"] = standardized

    # STEP 2: CLEANED_DATASET
    # 1) remove high-risk columns based on missingness/id-like/constant/duplicates
    df_work, dropped_info = _drop_high_risk_columns(df_work, target_col, report, missing_threshold=0.5)

    # 2) drop duplicate rows
    before_dups = len(df_work)
    df_work = df_work.drop_duplicates()
    report["duplicate_rows_removed"] = before_dups - len(df_work)

    # 3) recompute types for imputation
    col_types = _identify_column_types_for_preprocessing(df_work)

    # 4) handle missing values (adaptive)
    _impute_missing_adaptive(df_work, col_types, report)

    # 5) cap outliers for numeric columns
    col_types = _identify_column_types_for_preprocessing(df_work)
    _cap_outliers_iqr(df_work, col_types["numeric"], report)

    cleaned_df = df_work

    # Validation: no missing in cleaned dataset
    cleaned_missing = int(cleaned_df.isna().sum().sum())
    report["cleaned_missing_values_total"] = cleaned_missing
    report["cleaned_shape"] = cleaned_df.shape

    # Build a compact cleaning summary
    steps_summary: List[str] = []
    steps_summary.append(
        f"Dataset started with {original_shape[0]} rows and {original_shape[1]} columns."
    )
    if report.get("detected_target_column"):
        steps_summary.append(f"Detected target column: `{target_col}` (kept unchanged for modeling).")
    steps_summary.append("Standardized categorical text (trimmed whitespace, normalized spacing, lowercased).")

    dropped_total = report.get("dropped_columns_total", []) or []
    if dropped_total:
        steps_summary.append("Dropped high-risk columns: " + ", ".join(map(str, dropped_total[:20])) + ("" if len(dropped_total) <= 20 else "..."))

    if report.get("duplicate_rows_removed", 0) > 0:
        steps_summary.append(f"Removed {report['duplicate_rows_removed']} duplicate rows.")

    imputed_cols = report.get("imputed_missing_columns") or []
    if imputed_cols:
        steps_summary.append("Imputed missing values using skew-adaptive mean/median (numeric) and mode (categorical/boolean).")

    if report.get("outlier_treatment_iqr"):
        steps_summary.append("Capped outliers using IQR capping for numeric columns.")

    steps_summary.append(
        f"Cleaned dataset has {cleaned_df.shape[0]} rows and {cleaned_df.shape[1]} columns."
    )
    cleaning_summary = "\n".join(f"{i+1}. {s}" for i, s in enumerate(steps_summary))

    # STEP 3: MODEL_READY_DATASET
    model_report: Dict[str, Any] = {}
    df_model = cleaned_df.copy()

    y = None
    if target_col and target_col in df_model.columns:
        y = df_model[target_col]
        X = df_model.drop(columns=[target_col])
    else:
        X = df_model

    # Datetime -> extract features
    datetime_cols = [c for c in X.columns if pd.api.types.is_datetime64_any_dtype(X[c])]
    for col in datetime_cols:
        dt = pd.to_datetime(X[col], errors="coerce")
        X[f"{col}_year"] = dt.dt.year
        X[f"{col}_month"] = dt.dt.month
        X[f"{col}_day"] = dt.dt.day
        X[f"{col}_day_of_week"] = dt.dt.dayofweek
    if datetime_cols:
        X = X.drop(columns=datetime_cols)
    model_report["datetime_extracted_columns"] = datetime_cols

    # Boolean -> 0/1
    bool_cols = [c for c in X.columns if X[c].dtype in ["bool", "boolean"] or pd.api.types.is_bool_dtype(X[c])]
    for col in bool_cols:
        # ===== BOOLEAN → INT (FIX) =====
        bool_cols = [c for c in X.columns if str(X[c].dtype) in ["bool", "boolean"]]

        for col in bool_cols:
            X[col] = X[col].astype(int)
    model_report["boolean_converted_columns"] = bool_cols

    # Identify remaining types
    categorical_cols = [c for c in X.columns if X[c].dtype == "object" or pd.api.types.is_categorical_dtype(X[c])]
    numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]

    # Encode categoricals
    one_hot_cols: List[str] = []
    label_encoded_cols: List[str] = []
    label_maps: Dict[str, Dict[str, int]] = {}

    encoded_parts: List[pd.DataFrame] = []
    for col in categorical_cols:
        nunique = X[col].nunique(dropna=True)
        if nunique <= 10:
            one_hot = pd.get_dummies(X[col], prefix=col, drop_first=True)
            one_hot_cols.append(col)
            encoded_parts.append(one_hot)
        else:
            # label encode with stable ordering
            series = X[col].astype(str)
            uniques = sorted(series.dropna().unique().tolist())
            mapping = {u: i for i, u in enumerate(uniques)}
            label_encoded = series.map(mapping).astype(float)
            label_encoded_cols.append(col)
            label_maps[col] = mapping
            encoded_parts.append(label_encoded.to_frame(col))

    X_num = X[numeric_cols].copy() if numeric_cols else pd.DataFrame(index=X.index)
    if encoded_parts:
        X_num = pd.concat([X_num] + encoded_parts, axis=1)

    # Ensure all feature columns numeric
    for col in X_num.columns:
        if not pd.api.types.is_numeric_dtype(X_num[col]):
            X_num[col] = pd.to_numeric(X_num[col], errors="coerce")

    # Fill any remaining missing in features with 0
    X_num = X_num.fillna(0)

    # Multicollinearity: remove features with abs(corr) > 0.9
    corr = None
    dropped_correlated: List[str] = []
    try:
        corr = X_num.corr().abs()
    except Exception:
        corr = None

    if corr is not None and not corr.empty:
        cols = list(corr.columns)
        kept: List[str] = []
        for c in cols:
            if not kept:
                kept.append(c)
                continue
            if any(corr.loc[c, k] > 0.9 for k in kept if k in corr.columns):
                dropped_correlated.append(c)
            else:
                kept.append(c)
        if dropped_correlated:
            X_num = X_num.drop(columns=dropped_correlated, errors="ignore")

    model_report["one_hot_encoded_columns"] = one_hot_cols
    model_report["label_encoded_columns"] = label_encoded_cols
    model_report["dropped_high_correlation_features"] = dropped_correlated
    model_report["label_encoding_maps_preview"] = {k: dict(list(v.items())[:10]) for k, v in label_maps.items()}

    # Scaling: StandardScaler-like z-score on feature columns only
    scaled_cols: List[str] = []
    for col in X_num.columns:
        mean = float(X_num[col].mean())
        std = float(X_num[col].std(ddof=0))
        if std == 0 or np.isnan(std):
            X_num[col] = 0.0
        else:
            X_num[col] = (X_num[col] - mean) / std
        scaled_cols.append(col)

    model_report["scaled_feature_columns"] = scaled_cols

    model_ready_df = X_num
    if y is not None:
        # Keep target unchanged (no scaling). Ensure it remains attached.
        model_ready_df[target_col] = y

    # Final validation
    final_missing = int(model_ready_df.isna().sum().sum())
    model_report["model_ready_missing_values_total"] = final_missing
    model_report["model_ready_shape"] = model_ready_df.shape

    # Ensure features (excluding target) are numeric and no booleans
    if target_col and target_col in model_ready_df.columns:
        features = model_ready_df.drop(columns=[target_col])
    else:
        features = model_ready_df
    # convert any boolean leftovers
    for col in features.columns:
        if pd.api.types.is_bool_dtype(features[col]):
            features[col] = features[col].astype(int)

    # no object columns in features
    for col in features.columns:
        if features[col].dtype == "object":
            features[col] = pd.to_numeric(features[col], errors="coerce")
    features = features.fillna(0)
    if target_col and target_col in model_ready_df.columns:
        model_ready_df = pd.concat([features, model_ready_df[[target_col]]], axis=1)
    else:
        model_ready_df = features

    # Recommended modeling features: all columns excluding target (if found)
    recommended_features = list(model_ready_df.columns)
    if target_col and target_col in model_ready_df.columns:
        recommended_features = [c for c in recommended_features if c != target_col]

    return {
        "cleaned_df": cleaned_df,
        "cleaning_summary": cleaning_summary,
        "recommended_features": recommended_features,
        "model_ready_df": model_ready_df,
        "metadata": {
            "target_column": target_col,
            "report": report,
            "model_report": model_report,
        },
    }

