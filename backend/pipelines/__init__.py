"""
Data Processing Pipelines Module

This module contains both legacy and core data processing pipelines for the AI Data Analyst Agent.

Core Pipelines (Recommended):
- enhanced_preprocess.py: Advanced preprocessing with enterprise features
- enhanced_data_cleaning.py: Advanced cleaning with comprehensive quality scoring

Legacy Pipelines:
- preprocess.py: Basic preprocessing functionality  
- data_cleaning.py: Basic data cleaning functionality

Usage:
# Core (recommended)
from backend.pipelines.core.enhanced_preprocess import preprocess_dataset_enhanced
from backend.pipelines.core.enhanced_data_cleaning import clean_dataset_enhanced

# Legacy (for backward compatibility)
from backend.pipelines.legacy.preprocess import preprocess_dataset
from backend.pipelines.legacy.data_cleaning import clean_dataset
"""

__version__ = "2.0.0"
