"""
Core Data Processing Pipelines (Enhanced)

This module contains the enhanced, enterprise-grade data processing pipelines
recommended for all new development.

Modules:
- enhanced_preprocess.py: Advanced preprocessing with multi-format support, quality scoring
- enhanced_data_cleaning.py: Advanced cleaning with pattern analysis, outlier detection

Features:
- International format support (US/European numbers, currencies, scientific notation)
- Advanced missing value analysis with pattern detection
- Multi-method outlier detection (IQR, Z-score, Isolation Forest)
- Comprehensive data quality scoring (A-F grades)
- Intelligent categorical encoding strategies
- Configurable processing parameters
- Detailed reporting and recommendations

Usage:
from backend.pipelines.core.enhanced_preprocess import preprocess_dataset_enhanced
from backend.pipelines.core.enhanced_data_cleaning import clean_dataset_enhanced

# Basic usage
cleaned = clean_dataset_enhanced(df)
model_ready = preprocess_dataset_enhanced(cleaned['cleaned_df'])

# With custom configuration
config = {"missing_threshold": 0.3, "scaling_method": "robust"}
result = preprocess_dataset_enhanced(df, config)
"""

from .enhanced_preprocess import preprocess_dataset_enhanced, EnhancedPreprocessor
from .enhanced_data_cleaning import clean_dataset_enhanced, EnhancedDataCleaner

__all__ = [
    'preprocess_dataset_enhanced',
    'EnhancedPreprocessor', 
    'clean_dataset_enhanced',
    'EnhancedDataCleaner'
]
