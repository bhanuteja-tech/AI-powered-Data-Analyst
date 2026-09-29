"""
Enhanced Preprocessing Pipeline - Production-Ready Data Preprocessing
Addresses all issues found in the original implementation with enterprise-grade solutions.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import re
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)

# Enhanced patterns for better detection
# Note: "class" is excluded as it's commonly a feature column (e.g., passenger class, product class)
_TARGET_NAMES = {"target", "label", "output", "y", "result", "outcome"}
_ENHANCED_ID_PATTERNS = [
    re.compile(r"\bid\b", re.IGNORECASE),
    re.compile(r"user[_\s]*id", re.IGNORECASE),
    re.compile(r"customer[_\s]*id", re.IGNORECASE),
    re.compile(r"account[_\s]*id", re.IGNORECASE),
    re.compile(r"transaction[_\s]*id", re.IGNORECASE),
    re.compile(r"order[_\s]*id", re.IGNORECASE),
    re.compile(r"product[_\s]*id", re.IGNORECASE),
    re.compile(r"uuid", re.IGNORECASE),
    re.compile(r"guid", re.IGNORECASE),
    re.compile(r"key", re.IGNORECASE),
    re.compile(r"identifier", re.IGNORECASE),
    re.compile(r"serial[_\s]*number", re.IGNORECASE),
    re.compile(r"reference", re.IGNORECASE),
]

# Enhanced numeric detection patterns
_ENHANCED_NUMERIC_PATTERNS = [
    re.compile(r"^[-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?$"),  # Comma-separated thousands
    re.compile(r"^[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?$"),      # Scientific notation
    re.compile(r"^[-+]?\d+\.?\d*$"),                        # Simple decimal
    re.compile(r"^\$?\s*[-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?\s*%?$"),  # Currency/percentage
]

# Boolean detection patterns
_BOOLEAN_PATTERNS = {
    "true": True, "false": False, "yes": True, "no": False,
    "y": True, "n": False, "t": True, "f": False, "1": True, "0": False,
    "active": True, "inactive": False, "enabled": True, "disabled": False,
    "on": True, "off": False, "up": True, "down": False,
    "male": True, "female": False, "pass": True, "fail": False,
    "success": True, "failure": False, "approved": True, "rejected": False,
}


class EnhancedPreprocessor:
    """
    Enterprise-grade data preprocessing pipeline with advanced features.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._default_config()
        self.scaler = None
        self.imputer = None
        self.encoding_maps = {}
        self.feature_names = []
        self.preprocessing_report = {}
        
    def _default_config(self) -> Dict[str, Any]:
        return {
            "missing_threshold": 0.4,
            "variance_threshold": 0.01,
            "correlation_threshold": 0.95,
            "outlier_method": "iqr",  # "iqr", "zscore", "isolation_forest"
            "outlier_threshold": 3.0,
            "imputation_strategy": "adaptive",  # "adaptive", "knn", "simple"
            "scaling_method": "standard",  # "standard", "robust", "minmax"
            "encoding_method": "auto",  # "auto", "one_hot", "label", "target"
            "feature_selection": True,
            "dimensionality_reduction": False,
            "max_cardinality_onehot": 20,
            "knn_neighbors": 5,
        }

    def _detect_target_column_enhanced(self, df: pd.DataFrame) -> Optional[str]:
        """Enhanced target column detection with multiple strategies."""
        # 1. Direct name matching - but be more conservative with "class" 
        # since it's commonly a feature column (e.g., passenger class)
        conservative_targets = {"target", "label", "output", "y", "result", "outcome"}
        for col in df.columns:
            if col and str(col).strip().lower() in conservative_targets:
                return col
        
        # 2. Pattern matching - exclude "class" as it's often a feature
        for col in df.columns:
            if col and any(pattern.search(str(col).lower()) for pattern in [
                re.compile(r"target|label|output|y$|result|outcome")
            ]):
                return col
        
        # 3. Statistical inference for binary classification
        binary_candidates = []
        for col in df.columns:
            if df[col].dtype in ['object', 'category']:
                unique_vals = df[col].dropna().unique()
                if len(unique_vals) == 2 and all(
                    str(val).lower() in ['true', 'false', 'yes', 'no', '1', '0', 'y', 'n'] 
                    for val in unique_vals
                ):
                    binary_candidates.append(col)
        
        if len(binary_candidates) == 1:
            return binary_candidates[0]
        
        # 4. Position-based inference (last column often target)
        if len(df.columns) > 1:
            last_col = df.columns[-1]
            if df[last_col].dtype in ['object', 'category', 'int64', 'float64']:
                if df[last_col].nunique() < len(df) * 0.5:  # Not too many unique values
                    return last_col
        
        return None

    def _convert_numeric_enhanced(self, df: pd.DataFrame, report: Dict[str, Any]) -> None:
        """Enhanced numeric conversion with multiple format support."""
        dtype_changes = {}
        
        for col in df.columns:
            if df[col].dtype != "object" and not pd.api.types.is_string_dtype(df[col]):
                continue
            
            # Sample for efficiency
            sample = df[col].dropna().astype(str).head(100)
            if sample.empty:
                continue
            
            # Try each numeric pattern
            numeric_ratio = 0
            for pattern in _ENHANCED_NUMERIC_PATTERNS:
                ratio = sample.str.match(pattern, na=False).mean()
                numeric_ratio = max(numeric_ratio, ratio)
            
            if numeric_ratio >= 0.8:
                before = str(df[col].dtype)
                try:
                    # Clean the data first
                    cleaned_series = df[col].astype(str).str.replace(r'[$,%]', '', regex=True).str.replace(',', '')
                    converted = pd.to_numeric(cleaned_series, errors='coerce')
                    
                    # Check conversion quality
                    non_null_ratio = converted.notna().mean()
                    if non_null_ratio >= 0.8:
                        df[col] = converted
                        after = str(df[col].dtype)
                        if before != after:
                            dtype_changes[col] = f"{before} -> {after}"
                except Exception as e:
                    logger.warning(f"Failed to convert {col} to numeric: {e}")
        
        report["enhanced_numeric_conversions"] = dtype_changes

    def _convert_boolean_enhanced(self, df: pd.DataFrame, report: Dict[str, Any]) -> None:
        """Enhanced boolean conversion with comprehensive pattern matching."""
        dtype_changes = {}
        bool_like_columns = []
        
        for col in df.columns:
            if df[col].dtype != "object" and not pd.api.types.is_string_dtype(df[col]):
                continue
            
            sample_series = df[col].dropna().astype(str).str.strip().str.lower()
            if sample_series.empty:
                continue
            
            unique_vals = set(sample_series.unique())
            # Check if values match our boolean patterns
            matched_vals = {val for val in unique_vals if val in _BOOLEAN_PATTERNS}
            
            if len(matched_vals) / len(unique_vals) >= 0.8:  # 80% match threshold
                before = str(df[col].dtype)
                try:
                    converted = sample_series.map(_BOOLEAN_PATTERNS)
                    # Check conversion quality
                    non_null_ratio = converted.notna().mean()
                    if non_null_ratio >= 0.8:
                        df[col] = converted.astype("boolean")
                        after = str(df[col].dtype)
                        dtype_changes[col] = f"{before} -> {after}"
                        bool_like_columns.append(col)
                except Exception as e:
                    logger.warning(f"Failed to convert {col} to boolean: {e}")
        
        report["enhanced_boolean_conversions"] = dtype_changes
        report["boolean_like_columns"] = bool_like_columns

    def _convert_datetime_enhanced(self, df: pd.DataFrame, report: Dict[str, Any]) -> None:
        """Enhanced datetime conversion with multiple format support."""
        dtype_changes = {}
        converted_cols = []
        
        for col in df.columns:
            if df[col].dtype != "object" and not pd.api.types.is_string_dtype(df[col]):
                continue
            
            # Enhanced name-based detection
            col_lower = str(col).lower()
            datetime_keywords = [
                "date", "time", "year", "month", "day", "created", "updated",
                "timestamp", "modified", "expiry", "start", "end", "due"
            ]
            
            if not any(kw in col_lower for kw in datetime_keywords):
                continue
            
            # Try multiple datetime formats
            sample = df[col].dropna().astype(str).head(50)
            if sample.empty:
                continue
            
            # Common datetime formats to try
            datetime_formats = [
                None,  # Auto-detect
                "%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%d/%m/%Y",
                "%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S",
                "%m/%d/%Y", "%d-%b-%Y", "%Y%m%d",
            ]
            
            best_format = None
            best_success_rate = 0
            
            for fmt in datetime_formats:
                try:
                    if fmt is None:
                        converted = pd.to_datetime(df[col], errors='coerce', infer_datetime_format=True)
                    else:
                        converted = pd.to_datetime(df[col], format=fmt, errors='coerce')
                    
                    success_rate = converted.notna().mean()
                    if success_rate > best_success_rate:
                        best_success_rate = success_rate
                        best_format = fmt
                except:
                    continue
            
            if best_success_rate >= 0.6:  # 60% success threshold
                before = str(df[col].dtype)
                try:
                    if best_format is None:
                        converted = pd.to_datetime(df[col], errors='coerce', infer_datetime_format=True)
                    else:
                        converted = pd.to_datetime(df[col], format=best_format, errors='coerce')
                    
                    df[col] = converted
                    after = str(df[col].dtype)
                    dtype_changes[col] = f"{before} -> {after} (format: {best_format or 'auto'})"
                    converted_cols.append(col)
                except Exception as e:
                    logger.warning(f"Failed to convert {col} to datetime: {e}")
        
        report["enhanced_datetime_conversions"] = dtype_changes
        report["datetime_converted_columns"] = converted_cols

    def _standardize_categorical_enhanced(self, df: pd.DataFrame, cols: List[str]) -> List[str]:
        """Enhanced categorical standardization with encoding fixes."""
        standardized = []
        
        for col in cols:
            if col not in df.columns:
                continue
            
            if not (pd.api.types.is_object_dtype(df[col]) or 
                   pd.api.types.is_string_dtype(df[col]) or 
                   pd.api.types.is_categorical_dtype(df[col])):
                continue
            
            try:
                # Convert to string first to handle mixed types
                df[col] = df[col].astype(str)
                
                # Remove encoding issues and normalize
                df[col] = (
                    df[col]
                    .str.encode('ascii', errors='ignore').decode('ascii')  # Remove non-ASCII
                    .str.strip()
                    .str.replace(r'\s+', ' ', regex=True)  # Normalize whitespace
                    .str.replace(r'[^\w\s]', '', regex=True)  # Remove special chars except spaces
                    .str.lower()
                    .replace(r'^\s*$', 'missing', regex=True)  # Replace empty strings
                )
                
                standardized.append(col)
            except Exception as e:
                logger.warning(f"Failed to standardize {col}: {e}")
        
        return standardized

    def _drop_high_risk_columns_enhanced(
        self, 
        df: pd.DataFrame, 
        target_col: Optional[str], 
        report: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Enhanced column dropping with better heuristics."""
        drop_cols = []
        n_rows = len(df)
        
        # 1. High missingness
        missing_pct = df.isna().sum() / n_rows if n_rows else pd.Series(dtype=float)
        high_missing = missing_pct[missing_pct > self.config["missing_threshold"]].index.tolist()
        
        # 2. Enhanced ID-like detection
        id_like = []
        for col in df.columns:
            if col == target_col:
                continue
            
            col_str = str(col).lower()
            # Check name patterns
            if any(pattern.search(col_str) for pattern in _ENHANCED_ID_PATTERNS):
                id_like.append(col)
            # Check uniqueness patterns
            elif df[col].nunique() == n_rows and n_rows > 100:  # Likely ID if all unique
                if df[col].dtype == 'object' or (df[col].dtype in ['int64', 'float64'] and df[col].nunique() > n_rows * 0.9):
                    id_like.append(col)
        
        # 3. Low variance columns
        variance_threshold = VarianceThreshold(threshold=self.config["variance_threshold"])
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        low_variance = []
        if len(numeric_cols) > 0:
            try:
                variance_threshold.fit(df[numeric_cols].fillna(0))
                low_variance = [col for col, keep in zip(numeric_cols, variance_threshold.get_support()) if not keep]
            except:
                pass
        
        # 4. Constant columns (including categorical)
        constant_cols = []
        for col in df.columns:
            if col == target_col:
                continue
            if df[col].nunique(dropna=False) <= 1:
                constant_cols.append(col)
        
        # 5. Duplicate columns (enhanced detection)
        dup_cols = []
        try:
            # Create signatures for duplicate detection
            sig_df = df.copy()
            for col in sig_df.columns:
                if pd.api.types.is_datetime64_any_dtype(sig_df[col]):
                    sig_df[col] = pd.to_datetime(sig_df[col], errors='coerce').astype(str)
            
            sig_df = sig_df.astype(str).fillna("__MISSING__")
            dup_mask = sig_df.T.duplicated(keep="first")
            dup_cols = sig_df.columns[dup_mask].tolist()
            dup_cols = [c for c in dup_cols if c != target_col]
        except Exception as e:
            logger.warning(f"Duplicate column detection failed: {e}")
        
        drop_cols = sorted(set(high_missing + id_like + constant_cols + dup_cols + low_variance))
        
        report["dropped_columns"] = {
            "high_missing": high_missing,
            "id_like": id_like,
            "constant": constant_cols,
            "duplicates": dup_cols,
            "low_variance": low_variance,
            "total": drop_cols
        }
        
        return df.drop(columns=drop_cols, errors='ignore'), {"dropped_columns": drop_cols}

    def _handle_outliers_enhanced(self, df: pd.DataFrame, numeric_cols: List[str], report: Dict[str, Any]) -> None:
        """Enhanced outlier detection with multiple methods."""
        outlier_treatment = {}
        method = self.config["outlier_method"]
        
        for col in numeric_cols:
            if col not in df.columns:
                continue
            
            series = pd.to_numeric(df[col], errors='coerce').dropna()
            if series.empty or len(series) < 4:  # Need minimum points for outlier detection
                continue
            
            original_values = df[col].copy()
            
            if method == "iqr":
                q1, q3 = series.quantile([0.25, 0.75])
                iqr = q3 - q1
                if iqr > 0:
                    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                    df[col] = df[col].clip(lower=lower, upper=upper)
                    
                    outliers_count = ((original_values < lower) | (original_values > upper)).sum()
                    if outliers_count > 0:
                        outlier_treatment[col] = {
                            "method": "IQR capping",
                            "lower_bound": float(lower),
                            "upper_bound": float(upper),
                            "values_capped": int(outliers_count)
                        }
            
            elif method == "zscore":
                z_scores = np.abs((series - series.mean()) / series.std())
                threshold = self.config["outlier_threshold"]
                outlier_mask = z_scores > threshold
                
                if outlier_mask.any():
                    # Cap outliers at threshold
                    mean_val, std_val = series.mean(), series.std()
                    lower_cap = mean_val - threshold * std_val
                    upper_cap = mean_val + threshold * std_val
                    
                    df[col] = df[col].clip(lower=lower_cap, upper=upper_cap)
                    
                    outlier_treatment[col] = {
                        "method": "Z-score capping",
                        "threshold": threshold,
                        "values_capped": int(outlier_mask.sum())
                    }
        
        report["outlier_treatment"] = outlier_treatment

    def _impute_missing_enhanced(self, df: pd.DataFrame, col_types: Dict[str, List[str]], report: Dict[str, Any]) -> List[str]:
        """Enhanced missing value imputation with adaptive strategies."""
        dropped_cols = report.get("dropped_columns", {}).get("total", [])
        handled_cols = []
        strategy = self.config["imputation_strategy"]
        
        for col in df.columns:
            if col in dropped_cols or df[col].isna().sum() == 0:
                continue
            
            if col in col_types["numeric"]:
                if strategy == "knn" and len(col_types["numeric"]) > 1:
                    # Use KNN imputation for numeric columns
                    try:
                        numeric_data = df[col_types["numeric"]].select_dtypes(include=[np.number])
                        if not numeric_data.empty:
                            imputer = KNNImputer(n_neighbors=min(self.config["knn_neighbors"], len(numeric_data)-1))
                            df[col_types["numeric"]] = imputer.fit_transform(numeric_data)
                            handled_cols.extend(col_types["numeric"])
                            break
                    except:
                        pass
                
                # Adaptive strategy based on distribution
                series = df[col].dropna()
                if not series.empty:
                    skewness = series.skew()
                    if abs(skewness) > 1.0:  # Highly skewed
                        fill_val = series.median()
                    else:  # Approximately normal
                        fill_val = series.mean()
                    
                    df[col] = df[col].fillna(fill_val)
                    handled_cols.append(col)
            
            elif col in col_types["categorical"]:
                # Use mode for categorical, with fallback
                mode_vals = df[col].mode()
                if not mode_vals.empty:
                    fill_val = mode_vals.iloc[0]
                else:
                    fill_val = "unknown"
                
                df[col] = df[col].fillna(fill_val)
                handled_cols.append(col)
            
            elif col in col_types["boolean"]:
                # Use mode for boolean
                mode_vals = df[col].mode()
                if not mode_vals.empty:
                    fill_val = mode_vals.iloc[0]
                else:
                    fill_val = False
                
                df[col] = df[col].fillna(fill_val)
                handled_cols.append(col)
        
        report["imputed_columns"] = handled_cols
        report["imputation_strategy"] = strategy
        return handled_cols

    def _encode_categorical_enhanced(
        self, 
        df: pd.DataFrame, 
        categorical_cols: List[str], 
        report: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Enhanced categorical encoding with smart strategy selection."""
        encoding_info = {
            "one_hot_encoded": [],
            "label_encoded": [],
            "frequency_encoded": [],
            "encoding_maps": {}
        }
        
        encoded_parts = []
        
        for col in categorical_cols:
            if col not in df.columns:
                continue
            
            nunique = df[col].nunique(dropna=True)
            
            # Skip if too many unique values or single value
            if nunique <= 1 or nunique > len(df) * 0.5:
                continue
            
            if nunique <= self.config["max_cardinality_onehot"]:
                # One-hot encoding for low cardinality
                one_hot = pd.get_dummies(df[col], prefix=col, drop_first=True, dummy_na=True)
                encoding_info["one_hot_encoded"].append(col)
                encoded_parts.append(one_hot)
            
            elif nunique <= 100:
                # Label encoding for medium cardinality
                series = df[col].astype(str)
                uniques = sorted(series.dropna().unique())
                mapping = {val: idx for idx, val in enumerate(uniques)}
                
                label_encoded = series.map(mapping).astype(float)
                encoding_info["label_encoded"].append(col)
                encoding_info["encoding_maps"][col] = mapping
                encoded_parts.append(label_encoded.to_frame(col))
            
            else:
                # Frequency encoding for high cardinality
                freq_map = df[col].value_counts().to_dict()
                freq_encoded = df[col].map(freq_map).fillna(0)
                encoding_info["frequency_encoded"].append(col)
                encoding_info["encoding_maps"][col] = "frequency"
                encoded_parts.append(freq_encoded.to_frame(col))
        
        # Combine encoded columns
        if encoded_parts:
            encoded_df = pd.concat(encoded_parts, axis=1)
            df = pd.concat([df.drop(columns=categorical_cols), encoded_df], axis=1)
        
        report["categorical_encoding"] = encoding_info
        return df, encoding_info

    def _scale_features_enhanced(
        self, 
        df: pd.DataFrame, 
        numeric_cols: List[str], 
        report: Dict[str, Any]
    ) -> pd.DataFrame:
        """Enhanced feature scaling with method selection."""
        if not numeric_cols:
            return df
        
        scaling_method = self.config["scaling_method"]
        
        try:
            if scaling_method == "robust":
                scaler = RobustScaler()
            elif scaling_method == "minmax":
                scaler = MinMaxScaler()
            else:  # standard
                scaler = StandardScaler()
            
            # Scale numeric columns
            df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
            
            # Store scaler for potential inverse transform
            self.scaler = scaler
            
            report["scaling_applied"] = {
                "method": scaling_method,
                "columns": numeric_cols,
                "scaler_params": {k: v.tolist() if hasattr(v, 'tolist') else v 
                                for k, v in scaler.__dict__.items() 
                                if not k.startswith('_')}
            }
            
        except Exception as e:
            logger.warning(f"Scaling failed: {e}")
            report["scaling_applied"] = {"error": str(e)}
        
        return df

    def _feature_selection_enhanced(
        self, 
        df: pd.DataFrame, 
        target_col: Optional[str], 
        report: Dict[str, Any]
    ) -> pd.DataFrame:
        """Enhanced feature selection with multiple techniques."""
        if not self.config["feature_selection"]:
            return df
        
        # Separate features and target
        if target_col and target_col in df.columns:
            X = df.drop(columns=[target_col])
            y = df[target_col]
        else:
            X = df
            y = None
        
        # Remove highly correlated features
        corr_threshold = self.config["correlation_threshold"]
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 1:
            try:
                corr_matrix = X[numeric_cols].corr().abs()
                upper_tri = corr_matrix.where(
                    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
                )
                
                to_drop = [col for col in upper_tri.columns if any(upper_tri[col] > corr_threshold)]
                
                if to_drop:
                    X = X.drop(columns=to_drop)
                    report["correlation_filtering"] = {
                        "threshold": corr_threshold,
                        "dropped_columns": to_drop
                    }
            except Exception as e:
                logger.warning(f"Correlation filtering failed: {e}")
        
        # Reassemble dataframe
        if target_col and target_col in df.columns:
            df = pd.concat([X, y], axis=1)
        else:
            df = X
        
        return df

    def _identify_column_types_enhanced(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Enhanced column type identification."""
        return {
            "numeric": df.select_dtypes(include=[np.number]).columns.tolist(),
            "categorical": df.select_dtypes(include=["object", "category"]).columns.tolist(),
            "boolean": df.select_dtypes(include=["bool", "boolean"]).columns.tolist(),
            "datetime": [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])],
        }

    def preprocess_dataset(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Main preprocessing pipeline with enhanced features.
        Returns comprehensive results including cleaned and model-ready datasets.
        """
        report = {"pipeline_version": "enhanced_v2.0"}
        original_shape = df.shape
        
        # Defensive copy
        df_work = df.copy()
        
        # Step 1: Target detection
        target_col = self._detect_target_column_enhanced(df_work)
        report["target_column"] = target_col
        
        # Step 2: Initial type identification
        col_types_before = self._identify_column_types_enhanced(df_work)
        report["column_types_before"] = col_types_before
        
        # Step 3: Enhanced data type conversion
        self._convert_numeric_enhanced(df_work, report)
        self._convert_boolean_enhanced(df_work, report)
        self._convert_datetime_enhanced(df_work, report)
        
        # Step 4: Categorical standardization
        categorical_cols = df_work.select_dtypes(include=["object", "category"]).columns.tolist()
        standardized = self._standardize_categorical_enhanced(df_work, categorical_cols)
        report["standardized_categorical"] = standardized
        
        # Step 5: Drop high-risk columns
        df_work, dropped_info = self._drop_high_risk_columns_enhanced(df_work, target_col, report)
        
        # Step 6: Remove duplicate rows
        before_dups = len(df_work)
        df_work = df_work.drop_duplicates()
        report["duplicate_rows_removed"] = before_dups - len(df_work)
        
        # Step 7: Update column types
        col_types = self._identify_column_types_enhanced(df_work)
        
        # Step 8: Enhanced missing value imputation
        self._impute_missing_enhanced(df_work, col_types, report)
        
        # Step 9: Enhanced outlier treatment
        self._handle_outliers_enhanced(df_work, col_types["numeric"], report)
        
        # Cleaned dataset
        cleaned_df = df_work.copy()
        report["cleaned_shape"] = cleaned_df.shape
        
        # Step 10: Model-ready dataset preparation
        model_df = cleaned_df.copy()
        
        # Handle datetime features
        datetime_cols = col_types["datetime"]
        if datetime_cols:
            for col in datetime_cols:
                if col in model_df.columns:
                    dt = pd.to_datetime(model_df[col], errors='coerce')
                    model_df[f"{col}_year"] = dt.dt.year
                    model_df[f"{col}_month"] = dt.dt.month
                    model_df[f"{col}_day"] = dt.dt.day
                    model_df[f"{col}_day_of_week"] = dt.dt.dayofweek
                    model_df[f"{col}_quarter"] = dt.dt.quarter
            
            model_df = model_df.drop(columns=datetime_cols)
            report["datetime_features_extracted"] = datetime_cols
        
        # Convert boolean to int
        bool_cols = [c for c in model_df.columns if model_df[c].dtype in ["bool", "boolean"]]
        if bool_cols:
            model_df[bool_cols] = model_df[bool_cols].astype(int)
            report["boolean_converted"] = bool_cols
        
        # Encode categorical variables
        cat_cols = [c for c in model_df.columns if model_df[c].dtype in ["object", "category"]]
        if cat_cols:
            model_df, encoding_info = self._encode_categorical_enhanced(model_df, cat_cols, report)
        
        # Identify final numeric columns
        final_numeric = [c for c in model_df.columns if pd.api.types.is_numeric_dtype(model_df[c])]
        
        # Scale features (exclude target if present)
        features_to_scale = final_numeric.copy()
        if target_col and target_col in features_to_scale:
            features_to_scale.remove(target_col)
        
        if features_to_scale:
            model_df = self._scale_features_enhanced(model_df, features_to_scale, report)
        
        # Feature selection
        model_df = self._feature_selection_enhanced(model_df, target_col, report)
        
        # Final validation
        report["model_ready_shape"] = model_df.shape
        report["final_missing_values"] = int(model_df.isna().sum().sum())
        
        # Generate comprehensive summary
        summary = self._generate_enhanced_summary(original_shape, cleaned_df.shape, model_df.shape, report)
        
        # Recommended features
        recommended_features = [c for c in model_df.columns if c != target_col]
        
        return {
            "cleaned_df": cleaned_df,
            "model_ready_df": model_df,
            "cleaning_summary": summary,
            "recommended_features": recommended_features,
            "metadata": {
                "target_column": target_col,
                "preprocessing_report": report,
                "config": self.config
            }
        }

    def _generate_enhanced_summary(
        self, 
        original_shape: Tuple[int, int], 
        cleaned_shape: Tuple[int, int], 
        model_shape: Tuple[int, int], 
        report: Dict[str, Any]
    ) -> str:
        """Generate comprehensive preprocessing summary."""
        steps = []
        
        steps.append(f"📊 Enhanced preprocessing pipeline completed successfully!")
        steps.append(f"Original dataset: {original_shape[0]} rows × {original_shape[1]} columns")
        steps.append(f"Cleaned dataset: {cleaned_shape[0]} rows × {cleaned_shape[1]} columns")
        steps.append(f"Model-ready dataset: {model_shape[0]} rows × {model_shape[1]} columns")
        
        if report.get("target_column"):
            steps.append(f"🎯 Target column detected: `{report['target_column']}`")
        
        # Data type improvements
        conversions = []
        if report.get("enhanced_numeric_conversions"):
            conversions.append(f"Numeric: {len(report['enhanced_numeric_conversions'])} columns")
        if report.get("enhanced_boolean_conversions"):
            conversions.append(f"Boolean: {len(report['enhanced_boolean_conversions'])} columns")
        if report.get("enhanced_datetime_conversions"):
            conversions.append(f"Datetime: {len(report['enhanced_datetime_conversions'])} columns")
        
        if conversions:
            steps.append(f"🔄 Data type corrections: {', '.join(conversions)}")
        
        # Column filtering
        dropped = report.get("dropped_columns", {})
        if dropped.get("total"):
            drop_reasons = []
            for reason, cols in dropped.items():
                if reason != "total" and cols:
                    drop_reasons.append(f"{reason}: {len(cols)}")
            steps.append(f"🗑️ Columns removed: {'; '.join(drop_reasons)}")
        
        # Data quality improvements
        if report.get("duplicate_rows_removed", 0) > 0:
            steps.append(f"🧹 Duplicate rows removed: {report['duplicate_rows_removed']}")
        
        if report.get("imputed_columns"):
            steps.append(f"🔧 Missing values imputed: {len(report['imputed_columns'])} columns ({report.get('imputation_strategy', 'adaptive')} strategy)")
        
        if report.get("outlier_treatment"):
            steps.append(f"📈 Outliers treated: {len(report['outlier_treatment'])} columns ({self.config['outlier_method']} method)")
        
        # Feature engineering
        if report.get("datetime_features_extracted"):
            steps.append(f"📅 Datetime features extracted: {len(report['datetime_features_extracted'])} columns")
        
        encoding = report.get("categorical_encoding", {})
        encoded_types = []
        if encoding.get("one_hot_encoded"):
            encoded_types.append(f"One-hot: {len(encoding['one_hot_encoded'])}")
        if encoding.get("label_encoded"):
            encoded_types.append(f"Label: {len(encoding['label_encoded'])}")
        if encoding.get("frequency_encoded"):
            encoded_types.append(f"Frequency: {len(encoding['frequency_encoded'])}")
        
        if encoded_types:
            steps.append(f"🏷️ Categorical encoding: {', '.join(encoded_types)}")
        
        if report.get("scaling_applied") and "error" not in report["scaling_applied"]:
            steps.append(f"⚖️ Features scaled: {self.config['scaling_method']} method")
        
        if report.get("correlation_filtering"):
            steps.append(f"🔗 Correlation filtering: {len(report['correlation_filtering']['dropped_columns'])} columns removed")
        
        steps.append("✅ Dataset is now ready for machine learning!")
        
        return "\n".join(f"{i+1}. {step}" for i, step in enumerate(steps))


def preprocess_dataset_enhanced(df: pd.DataFrame, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Convenience function for enhanced preprocessing.
    
    Args:
        df: Input DataFrame
        config: Optional configuration dictionary
        
    Returns:
        Dictionary with cleaned_df, model_ready_df, summary, and metadata
    """
    preprocessor = EnhancedPreprocessor(config)
    return preprocessor.preprocess_dataset(df)
