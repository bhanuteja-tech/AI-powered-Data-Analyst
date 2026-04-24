"""
Enhanced Data Cleaning Pipeline - Enterprise-Grade Data Cleaning
Addresses all issues found in the original implementation with advanced techniques.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union
import re
import warnings
from datetime import datetime
import logging
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import chardet

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)


class EnhancedDataCleaner:
    """
    Enterprise-grade data cleaning pipeline with advanced features.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._default_config()
        self.cleaning_report = {}
        
    def _default_config(self) -> Dict[str, Any]:
        return {
            "missing_threshold": 0.4,
            "outlier_detection_methods": ["iqr", "zscore", "isolation_forest"],
            "outlier_threshold": 3.0,
            "duplicate_detection": "exact",  # "exact", "fuzzy"
            "text_cleaning": True,
            "encoding_detection": True,
            "advanced_imputation": True,
            "validation_rules": True,
            "profiling_depth": "comprehensive",  # "basic", "standard", "comprehensive"
            "correlation_analysis": True,
            "data_quality_score": True,
        }

    def _detect_file_encoding(self, file_path: str) -> str:
        """Detect file encoding for better text handling."""
        try:
            with open(file_path, 'rb') as f:
                raw_data = f.read(10000)  # Read first 10KB
                result = chardet.detect(raw_data)
                return result.get('encoding', 'utf-8')
        except:
            return 'utf-8'

    def _identify_column_types_enhanced(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Enhanced column type identification with better heuristics."""
        column_types = {
            "numeric": [],
            "categorical": [],
            "datetime": [],
            "boolean": [],
            "text": [],
            "mixed": []
        }
        
        for col in df.columns:
            series = df[col]
            
            # Skip completely empty columns
            if series.isna().all():
                continue
            
            # Boolean detection
            if pd.api.types.is_bool_dtype(series):
                column_types["boolean"].append(col)
                continue
            
            # Numeric detection
            if pd.api.types.is_numeric_dtype(series):
                # Check if it's actually categorical (like IDs stored as numbers)
                unique_ratio = series.nunique() / len(series.dropna())
                if unique_ratio > 0.5 and series.nunique() > 10:
                    # Likely numeric, but could be ID-like
                    if any(keyword in str(col).lower() for keyword in ['id', 'code', 'number']):
                        column_types["mixed"].append(col)
                    else:
                        column_types["numeric"].append(col)
                else:
                    # Low cardinality numeric - likely categorical
                    column_types["categorical"].append(col)
                continue
            
            # Datetime detection
            if pd.api.types.is_datetime64_any_dtype(series):
                column_types["datetime"].append(col)
                continue
            
            # Text/Categorical detection
            if pd.api.types.is_string_dtype(series) or series.dtype == 'object':
                # Sample for analysis
                sample = series.dropna().astype(str).head(100)
                
                if sample.empty:
                    continue
                
                # Check if it looks like datetime
                if any(keyword in str(col).lower() for keyword in 
                      ['date', 'time', 'year', 'month', 'day', 'created', 'updated']):
                    # Try to parse as datetime
                    try:
                        pd.to_datetime(sample.head(10), errors='raise')
                        column_types["datetime"].append(col)
                        continue
                    except:
                        pass
                
                # Check cardinality
                unique_ratio = sample.nunique() / len(sample)
                avg_length = sample.str.len().mean()
                
                # High cardinality text with long strings -> text column
                if unique_ratio > 0.7 and avg_length > 50:
                    column_types["text"].append(col)
                # Low cardinality -> categorical
                elif unique_ratio < 0.5 or sample.nunique() <= 20:
                    column_types["categorical"].append(col)
                else:
                    # Mixed characteristics
                    column_types["mixed"].append(col)
        
        return column_types

    def _enhanced_numeric_conversion(self, df: pd.DataFrame, report: Dict[str, Any]) -> pd.DataFrame:
        """Enhanced numeric conversion with multiple format support."""
        conversions = {}
        
        # Enhanced regex patterns for different numeric formats
        patterns = {
            'comma_decimal': re.compile(r'^-?\d{1,3}(?:\.\d{3})*,\d+$'),  # European: 1.234,56
            'comma_thousands': re.compile(r'^-?\$?\s*\d{1,3}(?:,\d{3})*(?:\.\d+)?\s*%?$'),  # US: 1,234.56
            'scientific': re.compile(r'^-?\d*\.?\d+(?:[eE][-+]?\d+)?$'),  # Scientific notation
            'currency': re.compile(r'^\$?\s*-?\d+(?:,\d{3})*(?:\.\d+)?\s*%?$'),  # Currency
            'percentage': re.compile(r'^-?\d+(?:,\d{3})*(?:\.\d+)?\s*%?$'),  # Percentage
            'simple': re.compile(r'^-?\d+\.?\d*$'),  # Simple decimal
        }
        
        for col in df.columns:
            if df[col].dtype != 'object' and not pd.api.types.is_string_dtype(df[col]):
                continue
            
            # Sample for analysis
            sample = df[col].dropna().astype(str).head(50)
            if sample.empty:
                continue
            
            best_match = None
            best_ratio = 0
            
            # Test each pattern
            for pattern_name, pattern in patterns.items():
                match_ratio = sample.str.match(pattern, na=False).mean()
                if match_ratio > best_ratio:
                    best_ratio = match_ratio
                    best_match = pattern_name
            
            if best_ratio >= 0.8:  # 80% threshold
                try:
                    cleaned_series = df[col].astype(str)
                    
                    # Apply specific cleaning based on pattern
                    if best_match == 'comma_decimal':
                        # Convert European format: 1.234,56 -> 1234.56
                        cleaned_series = cleaned_series.str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
                    elif best_match in ['comma_thousands', 'currency']:
                        # Remove currency symbols and commas
                        cleaned_series = cleaned_series.str.replace(r'[$,%]', '', regex=True)
                    elif best_match == 'percentage':
                        # Remove percentage sign
                        cleaned_series = cleaned_series.str.replace('%', '', regex=False)
                    
                    # Convert to numeric
                    converted = pd.to_numeric(cleaned_series, errors='coerce')
                    
                    # Check conversion quality
                    success_rate = converted.notna().mean()
                    if success_rate >= 0.8:
                        before_dtype = str(df[col].dtype)
                        df[col] = converted
                        conversions[col] = {
                            'from': before_dtype,
                            'to': str(df[col].dtype),
                            'pattern': best_match,
                            'success_rate': success_rate
                        }
                
                except Exception as e:
                    logger.warning(f"Failed to convert {col} using {best_match}: {e}")
        
        report['numeric_conversions'] = conversions
        return df

    def _enhanced_boolean_conversion(self, df: pd.DataFrame, report: Dict[str, Any]) -> pd.DataFrame:
        """Enhanced boolean conversion with comprehensive patterns."""
        conversions = {}
        
        # Extended boolean mappings
        boolean_mappings = {
            # Standard
            'true': True, 'false': False, 'yes': True, 'no': False,
            'y': True, 'n': False, 't': True, 'f': False,
            '1': True, '0': False,
            # Extended
            'active': True, 'inactive': False, 'enabled': True, 'disabled': False,
            'on': True, 'off': False, 'up': True, 'down': False,
            'male': True, 'female': False, 'pass': True, 'fail': False,
            'success': True, 'failure': False, 'approved': True, 'rejected': False,
            'complete': True, 'incomplete': False, 'valid': True, 'invalid': False,
            'agree': True, 'disagree': False, 'accept': True, 'decline': False,
            'high': True, 'low': False, 'positive': True, 'negative': False,
        }
        
        for col in df.columns:
            if df[col].dtype != 'object' and not pd.api.types.is_string_dtype(df[col]):
                continue
            
            sample = df[col].dropna().astype(str).str.strip().str.lower()
            if sample.empty:
                continue
            
            unique_vals = set(sample.unique())
            
            # Check if values match our boolean patterns
            matched_vals = {val for val in unique_vals if val in boolean_mappings}
            
            # Also check for partial matches (e.g., "TRUE", "False ")
            partial_matches = {val for val in unique_vals 
                             if any(val.startswith(key) or val.endswith(key) for key in boolean_mappings.keys())}
            
            total_matches = len(matched_vals.union(partial_matches))
            
            if total_matches / len(unique_vals) >= 0.8:  # 80% threshold
                try:
                    # Create mapping with partial matches
                    enhanced_mapping = boolean_mappings.copy()
                    for val in unique_vals:
                        val_lower = val.lower()
                        if val_lower not in enhanced_mapping:
                            # Check for partial matches
                            for key, bool_val in boolean_mappings.items():
                                if val_lower.startswith(key) or val_lower.endswith(key):
                                    enhanced_mapping[val_lower] = bool_val
                                    break
                    
                    converted = sample.map(enhanced_mapping)
                    success_rate = converted.notna().mean()
                    
                    if success_rate >= 0.8:
                        before_dtype = str(df[col].dtype)
                        df[col] = converted.astype('boolean')
                        conversions[col] = {
                            'from': before_dtype,
                            'to': 'boolean',
                            'unique_values': list(unique_vals),
                            'success_rate': success_rate
                        }
                
                except Exception as e:
                    logger.warning(f"Failed to convert {col} to boolean: {e}")
        
        report['boolean_conversions'] = conversions
        return df

    def _enhanced_datetime_conversion(self, df: pd.DataFrame, report: Dict[str, Any]) -> pd.DataFrame:
        """Enhanced datetime conversion with multiple format detection."""
        conversions = {}
        
        # Comprehensive datetime format list
        datetime_formats = [
            # ISO formats
            '%Y-%m-%d', '%Y/%m/%d', '%d-%m-%Y', '%d/%m/%Y',
            '%Y-%m-%d %H:%M:%S', '%Y/%m/%d %H:%M:%S', '%d-%m-%Y %H:%M:%S', '%d/%m/%Y %H:%M:%S',
            # US formats
            '%m/%d/%Y', '%m-%d-%Y', '%m/%d/%Y %H:%M:%S',
            # European formats
            '%d.%m.%Y', '%d.%m.%Y %H:%M:%S',
            # Text formats
            '%d-%b-%Y', '%d-%B-%Y', '%b %d, %Y', '%B %d, %Y',
            # Custom formats
            '%Y%m%d', '%Y%m%d%H%M%S', '%Y-%m-%d %H:%M', '%d-%m-%Y %H:%M',
            # Time zones
            '%Y-%m-%dT%H:%M:%S', '%Y-%m-%dT%H:%M:%SZ',
        ]
        
        # Enhanced column name detection
        datetime_keywords = [
            'date', 'time', 'year', 'month', 'day', 'created', 'updated', 'modified',
            'timestamp', 'expiry', 'start', 'end', 'due', 'birth', 'join', 'access',
            'last', 'first', 'at', 'on', 'from', 'to'
        ]
        
        for col in df.columns:
            if df[col].dtype != 'object' and not pd.api.types.is_string_dtype(df[col]):
                continue
            
            # Check column name
            col_lower = str(col).lower()
            name_match = any(keyword in col_lower for keyword in datetime_keywords)
            
            # Sample for testing
            sample = df[col].dropna().astype(str).head(20)
            if sample.empty:
                continue
            
            best_format = None
            best_success_rate = 0
            
            # Try each format
            for fmt in datetime_formats:
                try:
                    test_converted = pd.to_datetime(sample, format=fmt, errors='coerce')
                    success_rate = test_converted.notna().mean()
                    
                    if success_rate > best_success_rate:
                        best_success_rate = success_rate
                        best_format = fmt
                except:
                    continue
            
            # Also try auto-detection
            try:
                auto_converted = pd.to_datetime(sample, errors='coerce', infer_datetime_format=True)
                auto_success_rate = auto_converted.notna().mean()
                
                if auto_success_rate > best_success_rate:
                    best_success_rate = auto_success_rate
                    best_format = 'auto'
            except:
                pass
            
            # Decide if conversion is worthwhile
            threshold = 0.6 if name_match else 0.8  # Lower threshold if name suggests datetime
            
            if best_success_rate >= threshold:
                try:
                    if best_format == 'auto':
                        converted = pd.to_datetime(df[col], errors='coerce', infer_datetime_format=True)
                    else:
                        converted = pd.to_datetime(df[col], format=best_format, errors='coerce')
                    
                    final_success_rate = converted.notna().mean()
                    
                    if final_success_rate >= threshold:
                        before_dtype = str(df[col].dtype)
                        df[col] = converted
                        conversions[col] = {
                            'from': before_dtype,
                            'to': 'datetime64[ns]',
                            'format': best_format,
                            'success_rate': final_success_rate,
                            'name_match': name_match
                        }
                
                except Exception as e:
                    logger.warning(f"Failed to convert {col} to datetime: {e}")
        
        report['datetime_conversions'] = conversions
        return df

    def _advanced_text_cleaning(self, df: pd.DataFrame, text_cols: List[str], report: Dict[str, Any]) -> pd.DataFrame:
        """Advanced text cleaning for text columns."""
        cleaning_info = {}
        
        for col in text_cols:
            if col not in df.columns:
                continue
            
            original_series = df[col].copy()
            
            try:
                # Convert to string
                df[col] = df[col].astype(str)
                
                # Remove encoding issues
                df[col] = df[col].str.encode('ascii', errors='ignore').decode('ascii')
                
                # Normalize whitespace
                df[col] = df[col].str.replace(r'\s+', ' ', regex=True).str.strip()
                
                # Remove special characters but keep basic punctuation
                df[col] = df[col].str.replace(r'[^\w\s\.\,\!\?\;\:\-]', '', regex=True)
                
                # Handle common data quality issues
                df[col] = df[col].replace({
                    r'^\s*$': 'EMPTY',
                    'null': 'NULL',
                    'n/a': 'N/A',
                    'na': 'N/A',
                    'none': 'NONE',
                    'undefined': 'UNDEFINED'
                }, regex=True)
                
                # Calculate cleaning statistics
                changes = (original_series != df[col]).sum()
                empty_before = (original_series.str.strip() == '').sum()
                empty_after = (df[col] == 'EMPTY').sum()
                
                cleaning_info[col] = {
                    'values_changed': int(changes),
                    'empty_before': int(empty_before),
                    'empty_after': int(empty_after),
                    'unique_before': int(original_series.nunique()),
                    'unique_after': int(df[col].nunique())
                }
            
            except Exception as e:
                logger.warning(f"Text cleaning failed for {col}: {e}")
                cleaning_info[col] = {'error': str(e)}
        
        report['text_cleaning'] = cleaning_info
        return df

    def _enhanced_missing_value_analysis(self, df: pd.DataFrame, report: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced missing value analysis with patterns detection."""
        missing_analysis = {}
        
        # Basic missing statistics
        missing_counts = df.isnull().sum()
        missing_percentages = (missing_counts / len(df)) * 100
        
        # Missing patterns
        missing_patterns = {}
        for col in df.columns:
            if missing_counts[col] > 0:
                # Check if missing is systematic (e.g., all missing for certain conditions)
                series = df[col]
                
                # Check for completely missing rows
                completely_missing_rows = df[df[col].isna()].index.tolist()
                
                # Check correlation with other columns' missingness
                missing_correlation = {}
                for other_col in df.columns:
                    if other_col != col and missing_counts[other_col] > 0:
                        correlation = df[col].isna().corr(df[other_col].isna())
                        if not np.isnan(correlation) and abs(correlation) > 0.5:
                            missing_correlation[other_col] = correlation
                
                missing_patterns[col] = {
                    'count': int(missing_counts[col]),
                    'percentage': float(missing_percentages[col]),
                    'completely_missing_rows': len(completely_missing_rows),
                    'missing_correlation': missing_correlation
                }
        
        # Missing value patterns across rows
        row_missing_counts = df.isnull().sum(axis=1)
        missing_analysis['row_missing_patterns'] = {
            'rows_no_missing': int((row_missing_counts == 0).sum()),
            'rows_some_missing': int(((row_missing_counts > 0) & (row_missing_counts < len(df.columns))).sum()),
            'rows_mostly_missing': int((row_missing_counts >= len(df.columns) * 0.5).sum()),
            'avg_missing_per_row': float(row_missing_counts.mean()),
            'max_missing_per_row': int(row_missing_counts.max())
        }
        
        missing_analysis['column_patterns'] = missing_patterns
        
        # Recommendations
        high_missing_cols = [col for col, info in missing_patterns.items() 
                           if info['percentage'] > self.config['missing_threshold'] * 100]
        
        missing_analysis['recommendations'] = {
            'drop_columns': high_missing_cols,
            'impute_columns': [col for col in missing_patterns.keys() if col not in high_missing_cols],
            'investigate_patterns': [col for col, info in missing_patterns.items() 
                                   if len(info['missing_correlation']) > 0]
        }
        
        report['missing_value_analysis'] = missing_analysis
        return missing_analysis

    def _advanced_outlier_detection(self, df: pd.DataFrame, numeric_cols: List[str], report: Dict[str, Any]) -> Dict[str, Any]:
        """Advanced outlier detection using multiple methods."""
        outlier_results = {}
        
        for col in numeric_cols:
            if col not in df.columns:
                continue
            
            series = pd.to_numeric(df[col], errors='coerce').dropna()
            if len(series) < 4:  # Need minimum points
                continue
            
            col_results = {}
            
            # Method 1: IQR
            q1, q3 = series.quantile([0.25, 0.75])
            iqr = q3 - q1
            if iqr > 0:
                lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                iqr_outliers = (series < lower) | (series > upper)
                col_results['iqr'] = {
                    'count': int(iqr_outliers.sum()),
                    'percentage': float(iqr_outliers.mean() * 100),
                    'bounds': [float(lower), float(upper)]
                }
            
            # Method 2: Z-score
            if len(series) > 2:
                z_scores = np.abs(stats.zscore(series))
                z_outliers = z_scores > self.config['outlier_threshold']
                col_results['zscore'] = {
                    'count': int(z_outliers.sum()),
                    'percentage': float(z_outliers.mean() * 100),
                    'threshold': self.config['outlier_threshold']
                }
            
            # Method 3: Isolation Forest (for larger datasets)
            if len(series) > 30:
                try:
                    iso_forest = IsolationForest(contamination=0.1, random_state=42)
                    outlier_labels = iso_forest.fit_predict(series.values.reshape(-1, 1))
                    iso_outliers = outlier_labels == -1
                    col_results['isolation_forest'] = {
                        'count': int(iso_outliers.sum()),
                        'percentage': float(iso_outliers.mean() * 100),
                        'contamination': 0.1
                    }
                except Exception as e:
                    logger.warning(f"Isolation Forest failed for {col}: {e}")
            
            # Consensus: mark as outlier if detected by multiple methods
            if col_results:
                outlier_methods = list(col_results.keys())
                if len(outlier_methods) >= 2:
                    # Simple consensus: outlier if detected by at least 2 methods
                    outlier_masks = []
                    for method in outlier_methods:
                        if method == 'iqr':
                            outlier_masks.append(iqr_outliers)
                        elif method == 'zscore':
                            outlier_masks.append(z_outliers)
                        elif method == 'isolation_forest':
                            outlier_masks.append(iso_outliers)
                    
                    if outlier_masks:
                        consensus_outliers = np.sum(outlier_masks, axis=0) >= 2
                        col_results['consensus'] = {
                            'count': int(consensus_outliers.sum()),
                            'percentage': float(consensus_outliers.mean() * 100),
                            'methods_considered': outlier_methods
                        }
            
            outlier_results[col] = col_results
        
        report['outlier_detection'] = outlier_results
        return outlier_results

    def _handle_missing_values_advanced(self, df: pd.DataFrame, missing_analysis: Dict[str, Any], report: Dict[str, Any]) -> pd.DataFrame:
        """Advanced missing value handling with multiple strategies."""
        handling_info = {}
        
        # Drop columns with high missingness
        cols_to_drop = missing_analysis['recommendations']['drop_columns']
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
            handling_info['dropped_columns'] = cols_to_drop
        
        # Impute remaining columns
        cols_to_impute = missing_analysis['recommendations']['impute_columns']
        
        for col in cols_to_impute:
            if col not in df.columns:
                continue
            
            if pd.api.types.is_numeric_dtype(df[col]):
                # Advanced numeric imputation
                series = df[col].dropna()
                if not series.empty:
                    # Check distribution
                    skewness = stats.skew(series)
                    
                    if abs(skewness) > 1.0:
                        # Highly skewed - use median
                        fill_value = series.median()
                        method = 'median (skewed)'
                    else:
                        # Approximately normal - use mean
                        fill_value = series.mean()
                        method = 'mean (normal)'
                    
                    df[col] = df[col].fillna(fill_value)
                    handling_info[col] = {'method': method, 'value': float(fill_value)}
            
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                # Datetime imputation
                if df[col].notna().any():
                    # Use forward fill, then backward fill, then mode
                    df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
                    if df[col].isna().any():
                        mode_val = df[col].mode()
                        if not mode_val.empty:
                            df[col] = df[col].fillna(mode_val.iloc[0])
                            method = 'mode datetime'
                        else:
                            df[col] = df[col].fillna(pd.Timestamp.now())
                            method = 'current datetime'
                    else:
                        method = 'forward/backward fill'
                    
                    handling_info[col] = {'method': method}
            
            else:
                # Categorical/text imputation
                mode_vals = df[col].mode()
                if not mode_vals.empty:
                    fill_value = mode_vals.iloc[0]
                    method = 'mode'
                else:
                    fill_value = 'unknown'
                    method = 'unknown'
                
                df[col] = df[col].fillna(fill_value)
                handling_info[col] = {'method': method, 'value': str(fill_value)}
        
        report['missing_value_handling'] = handling_info
        return df

    def _handle_outliers_advanced(self, df: pd.DataFrame, outlier_results: Dict[str, Any], report: Dict[str, Any]) -> pd.DataFrame:
        """Advanced outlier handling with multiple strategies."""
        handling_info = {}
        
        for col, results in outlier_results.items():
            if col not in df.columns:
                continue
            
            if 'consensus' in results:
                # Use consensus outliers
                outlier_mask = self._get_outlier_mask(df, col, 'consensus', results)
            elif 'iqr' in results:
                # Fall back to IQR
                outlier_mask = self._get_outlier_mask(df, col, 'iqr', results)
            else:
                continue
            
            if outlier_mask.any():
                # Strategy: cap outliers instead of removing
                series = pd.to_numeric(df[col], errors='coerce')
                
                if 'iqr' in results:
                    bounds = results['iqr']['bounds']
                    df[col] = series.clip(lower=bounds[0], upper=bounds[1])
                    method = 'IQR capping'
                elif 'zscore' in results:
                    # Use z-score bounds
                    mean_val = series.mean()
                    std_val = series.std()
                    threshold = results['zscore']['threshold']
                    lower = mean_val - threshold * std_val
                    upper = mean_val + threshold * std_val
                    df[col] = series.clip(lower=lower, upper=upper)
                    method = 'Z-score capping'
                else:
                    method = 'no action'
                
                handling_info[col] = {
                    'method': method,
                    'outliers_handled': int(outlier_mask.sum())
                }
        
        report['outlier_handling'] = handling_info
        return df

    def _get_outlier_mask(self, df: pd.DataFrame, col: str, method: str, results: Dict[str, Any]) -> pd.Series:
        """Helper to get outlier mask for a specific method."""
        series = pd.to_numeric(df[col], errors='coerce')
        
        if method == 'iqr' and 'iqr' in results:
            bounds = results['iqr']['bounds']
            return (series < bounds[0]) | (series > bounds[1])
        elif method == 'zscore' and 'zscore' in results:
            z_scores = np.abs(stats.zscore(series.dropna()))
            return z_scores > results['zscore']['threshold']
        elif method == 'consensus' and 'consensus' in results:
            # Reconstruct consensus mask
            outlier_masks = []
            if 'iqr' in results:
                bounds = results['iqr']['bounds']
                outlier_masks.append((series < bounds[0]) | (series > bounds[1]))
            if 'zscore' in results:
                z_scores = np.abs(stats.zscore(series.dropna()))
                outlier_masks.append(z_scores > results['zscore']['threshold'])
            if 'isolation_forest' in results:
                iso_forest = IsolationForest(contamination=0.1, random_state=42)
                outlier_labels = iso_forest.fit_predict(series.values.reshape(-1, 1))
                outlier_masks.append(outlier_labels == -1)
            
            if outlier_masks:
                return pd.Series(np.sum(outlier_masks, axis=0) >= 2, index=series.index)
        
        return pd.Series(False, index=series.index)

    def _enhanced_duplicate_detection(self, df: pd.DataFrame, report: Dict[str, Any]) -> pd.DataFrame:
        """Enhanced duplicate detection with fuzzy matching options."""
        duplicate_info = {}
        
        # Exact duplicates
        before_count = len(df)
        exact_duplicates = df.duplicated()
        exact_duplicate_count = exact_duplicates.sum()
        
        if exact_duplicate_count > 0:
            df = df[~exact_duplicates]
            duplicate_info['exact_duplicates'] = {
                'count': int(exact_duplicate_count),
                'percentage': float(exact_duplicate_count / before_count * 100)
            }
        
        # Near duplicates for numeric columns (optional, computationally expensive)
        if self.config['duplicate_detection'] == 'fuzzy' and len(df) > 1:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                try:
                    # Use rounded values to find near duplicates
                    df_rounded = df[numeric_cols].round(2)
                    near_duplicates = df_rounded.duplicated()
                    near_duplicate_count = near_duplicates.sum()
                    
                    if near_duplicate_count > 0:
                        duplicate_info['near_duplicates_numeric'] = {
                            'count': int(near_duplicate_count),
                            'percentage': float(near_duplicate_count / len(df) * 100),
                            'tolerance': '0.01 (2 decimal places)'
                        }
                except Exception as e:
                    logger.warning(f"Near duplicate detection failed: {e}")
        
        duplicate_info['final_row_count'] = len(df)
        duplicate_info['rows_removed'] = before_count - len(df)
        
        report['duplicate_detection'] = duplicate_info
        return df

    def _calculate_data_quality_score(self, df: pd.DataFrame, report: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive data quality score."""
        quality_metrics = {}
        
        # Completeness score
        completeness = (1 - df.isna().sum().sum() / (len(df) * len(df.columns))) * 100
        quality_metrics['completeness'] = completeness
        
        # Uniqueness score (for rows)
        uniqueness = (len(df.drop_duplicates()) / len(df)) * 100 if len(df) > 0 else 0
        quality_metrics['uniqueness'] = uniqueness
        
        # Consistency score (based on data type consistency)
        consistency_scores = []
        for col in df.columns:
            if df[col].dtype in ['object', 'category']:
                # For categorical, check if values are consistent
                unique_ratio = df[col].nunique() / len(df.dropna())
                consistency_scores.append(max(0, (1 - unique_ratio) * 100))
            else:
                # For numeric/datetime, check conversion success
                consistency_scores.append(100)  # Assume consistent if properly typed
        
        consistency = np.mean(consistency_scores) if consistency_scores else 100
        quality_metrics['consistency'] = consistency
        
        # Validity score (based on outlier treatment)
        total_outliers = sum(results.get('consensus', {}).get('count', 0) 
                           for results in report.get('outlier_detection', {}).values())
        validity = max(0, (1 - total_outliers / (len(df) * len(df.columns))) * 100) if len(df) > 0 else 100
        quality_metrics['validity'] = validity
        
        # Overall score
        overall_score = np.mean(list(quality_metrics.values()))
        quality_metrics['overall'] = overall_score
        
        # Quality grade
        if overall_score >= 90:
            grade = 'A (Excellent)'
        elif overall_score >= 80:
            grade = 'B (Good)'
        elif overall_score >= 70:
            grade = 'C (Fair)'
        elif overall_score >= 60:
            grade = 'D (Poor)'
        else:
            grade = 'F (Very Poor)'
        
        quality_metrics['grade'] = grade
        
        report['data_quality_score'] = quality_metrics
        return quality_metrics

    def clean_dataset(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Main enhanced data cleaning pipeline.
        Returns comprehensive cleaning results with detailed reporting.
        """
        logger.info("Starting enhanced data cleaning pipeline")
        
        report = {
            'pipeline_version': 'enhanced_v2.0',
            'config': self.config,
            'start_time': datetime.now().isoformat()
        }
        
        original_shape = df.shape
        report['original_shape'] = original_shape
        
        # Step 1: Enhanced column type identification
        column_types = self._identify_column_types_enhanced(df)
        report['column_types'] = column_types
        
        # Step 2: Enhanced data type conversions
        df = self._enhanced_numeric_conversion(df, report)
        df = self._enhanced_boolean_conversion(df, report)
        df = self._enhanced_datetime_conversion(df, report)
        
        # Step 3: Advanced text cleaning
        if column_types.get('text') and self.config['text_cleaning']:
            df = self._advanced_text_cleaning(df, column_types['text'], report)
        
        # Step 4: Enhanced missing value analysis
        missing_analysis = self._enhanced_missing_value_analysis(df, report)
        
        # Step 5: Advanced outlier detection
        if column_types.get('numeric'):
            outlier_results = self._advanced_outlier_detection(df, column_types['numeric'], report)
        
        # Step 6: Enhanced duplicate detection
        df = self._enhanced_duplicate_detection(df, report)
        
        # Step 7: Handle missing values
        df = self._handle_missing_values_advanced(df, missing_analysis, report)
        
        # Step 8: Handle outliers
        if column_types.get('numeric'):
            df = self._handle_outliers_advanced(df, outlier_results, report)
        
        # Step 9: Final validation
        final_shape = df.shape
        report['final_shape'] = final_shape
        
        # Step 10: Data quality scoring
        if self.config['data_quality_score']:
            quality_scores = self._calculate_data_quality_score(df, report)
        
        # Generate comprehensive summary
        summary = self._generate_enhanced_summary(report)
        
        # Feature recommendations
        recommendations = self._generate_feature_recommendations(df, column_types, report)
        
        report['end_time'] = datetime.now().isoformat()
        
        logger.info("Enhanced data cleaning pipeline completed")
        
        return {
            'cleaned_df': df,
            'cleaning_summary': summary,
            'recommended_features': recommendations,
            'detailed_steps': report,
            'data_quality_score': quality_scores if self.config['data_quality_score'] else None
        }

    def _generate_enhanced_summary(self, report: Dict[str, Any]) -> str:
        """Generate comprehensive cleaning summary."""
        steps = []
        
        steps.append("🔧 Enhanced Data Cleaning Pipeline Completed")
        steps.append(f"📊 Dataset: {report['original_shape'][0]} rows × {report['original_shape'][1]} columns → {report['final_shape'][0]} rows × {report['final_shape'][1]} columns")
        
        # Data type improvements
        conversions = []
        if report.get('numeric_conversions'):
            conversions.append(f"Numeric: {len(report['numeric_conversions'])} columns")
        if report.get('boolean_conversions'):
            conversions.append(f"Boolean: {len(report['boolean_conversions'])} columns")
        if report.get('datetime_conversions'):
            conversions.append(f"Datetime: {len(report['datetime_conversions'])} columns")
        
        if conversions:
            steps.append(f"🔄 Data types corrected: {', '.join(conversions)}")
        
        # Text cleaning
        if report.get('text_cleaning'):
            steps.append(f"🧹 Text columns cleaned: {len(report['text_cleaning'])} columns")
        
        # Missing value handling
        if report.get('missing_value_analysis'):
            missing_info = report['missing_value_analysis']['recommendations']
            if missing_info['drop_columns']:
                steps.append(f"🗑️ Dropped high-missing columns: {len(missing_info['drop_columns'])}")
            if missing_info['impute_columns']:
                steps.append(f"🔧 Imputed missing values: {len(missing_info['impute_columns'])} columns")
        
        # Outlier handling
        if report.get('outlier_handling'):
            outlier_cols = len(report['outlier_handling'])
            total_outliers = sum(info.get('outliers_handled', 0) for info in report['outlier_handling'].values())
            steps.append(f"📈 Outliers treated: {total_outliers} values in {outlier_cols} columns")
        
        # Duplicate handling
        if report.get('duplicate_detection', {}).get('rows_removed', 0) > 0:
            removed = report['duplicate_detection']['rows_removed']
            steps.append(f"🧹 Duplicate rows removed: {removed}")
        
        # Data quality score
        if report.get('data_quality_score'):
            score = report['data_quality_score']['overall']
            grade = report['data_quality_score']['grade']
            steps.append(f"⭐ Data quality score: {score:.1f}/100 ({grade})")
        
        steps.append("✅ Dataset is now clean and ready for analysis!")
        
        return "\n".join(f"{i+1}. {step}" for i, step in enumerate(steps))

    def _generate_feature_recommendations(
        self, 
        df: pd.DataFrame, 
        column_types: Dict[str, List[str]], 
        report: Dict[str, Any]
    ) -> List[str]:
        """Generate intelligent feature recommendations for modeling."""
        recommendations = []
        
        # Numeric features
        numeric_cols = [col for col in column_types.get('numeric', []) if col in df.columns]
        if numeric_cols:
            # Filter out ID-like columns
            good_numeric = []
            for col in numeric_cols:
                if not any(keyword in str(col).lower() for keyword in ['id', 'index', 'key']):
                    good_numeric.append(col)
            
            if good_numeric:
                recommendations.extend(good_numeric[:10])  # Limit to top 10
        
        # Categorical features (reasonable cardinality)
        categorical_cols = [col for col in column_types.get('categorical', []) if col in df.columns]
        if categorical_cols:
            good_categorical = []
            for col in categorical_cols:
                unique_ratio = df[col].nunique() / len(df.dropna())
                if 0.01 <= unique_ratio <= 0.5:  # Reasonable cardinality
                    good_categorical.append(col)
            
            if good_categorical:
                recommendations.extend(good_categorical[:8])  # Limit to top 8
        
        # Boolean features
        boolean_cols = [col for col in column_types.get('boolean', []) if col in df.columns]
        if boolean_cols:
            recommendations.extend(boolean_cols[:5])  # Limit to top 5
        
        return recommendations[:20]  # Total limit of 20 recommendations


def clean_dataset_enhanced(df: pd.DataFrame, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Convenience function for enhanced data cleaning.
    
    Args:
        df: Input DataFrame
        config: Optional configuration dictionary
        
    Returns:
        Dictionary with cleaned_df, summary, recommendations, and detailed report
    """
    cleaner = EnhancedDataCleaner(config)
    return cleaner.clean_dataset(df)
