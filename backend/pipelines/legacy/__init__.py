"""
Legacy Data Processing Pipelines

This module contains the original data processing pipelines for backward compatibility.
These are maintained for existing code that may depend on the original API.

Modules:
- preprocess.py: Basic preprocessing functionality
- data_cleaning.py: Basic data cleaning functionality

Note: For new development, use the enhanced pipelines in the 'core' module.

Usage:
from backend.pipelines.legacy.preprocess import preprocess_dataset
from backend.pipelines.legacy.data_cleaning import clean_dataset

# Basic usage
cleaned = clean_dataset(df)
model_ready = preprocess_dataset(cleaned['cleaned_df'])
"""

from .preprocess import preprocess_dataset
from .data_cleaning import clean_dataset

__all__ = [
    'preprocess_dataset',
    'clean_dataset'
]
