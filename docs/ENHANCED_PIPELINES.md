# Enhanced Data Processing Pipelines

This document describes the enhanced preprocessing and data cleaning pipelines that address critical issues found in the original implementation and provide enterprise-grade data processing capabilities.

## Overview

The enhanced pipelines (`enhanced_preprocess.py` and `enhanced_data_cleaning.py`) provide:

- **Advanced data type detection** with international format support
- **Comprehensive missing value analysis** with pattern detection
- **Multiple outlier detection methods** (IQR, Z-score, Isolation Forest)
- **Intelligent categorical encoding** with strategy selection
- **Data quality scoring** with detailed metrics
- **Configurable processing parameters**
- **Enhanced error handling and logging**

## Key Improvements Over Original Implementation

### 1. Data Type Conversion Issues Fixed

**Original Problems:**
- Limited numeric pattern recognition
- No support for international number formats
- Basic boolean detection
- Simple datetime conversion

**Enhanced Solutions:**
- Multiple numeric format patterns (US, European, scientific notation, currency)
- Comprehensive boolean mappings (20+ patterns)
- Advanced datetime format detection (15+ formats)
- Encoding issue detection and correction

### 2. Missing Value Handling Issues Fixed

**Original Problems:**
- Simple median/mode imputation
- No pattern analysis
- Limited missing value insights

**Enhanced Solutions:**
- Missing value pattern analysis and correlation detection
- Adaptive imputation strategies based on data distribution
- KNN imputation option for numeric data
- Comprehensive missing value reporting

### 3. Outlier Detection Issues Fixed

**Original Problems:**
- Only IQR method available
- No multivariate outlier detection
- Limited outlier handling options

**Enhanced Solutions:**
- Multiple detection methods (IQR, Z-score, Isolation Forest)
- Consensus-based outlier detection
- Configurable outlier handling strategies
- Detailed outlier reporting

### 4. Categorical Processing Issues Fixed

**Original Problems:**
- Basic text cleaning only
- Limited encoding strategies
- No cardinality-based encoding selection

**Enhanced Solutions:**
- Advanced text cleaning with encoding normalization
- Smart encoding strategy selection (one-hot, label, frequency)
- Cardinality-based encoding decisions
- Encoding map preservation for reproducibility

### 5. Feature Engineering Issues Fixed

**Original Problems:**
- Basic datetime feature extraction
- Limited feature selection
- No correlation-based filtering

**Enhanced Solutions:**
- Comprehensive datetime feature extraction
- Correlation-based feature filtering
- Variance threshold feature selection
- Optional dimensionality reduction

## Usage

### Basic Usage

```python
from backend.enhanced_data_cleaning import clean_dataset_enhanced
from backend.enhanced_preprocess import preprocess_dataset_enhanced

# Load your data
df = pd.read_csv('your_data.csv')

# Clean the data
cleaning_result = clean_dataset_enhanced(df)
cleaned_df = cleaning_result['cleaned_df']
print(cleaning_result['cleaning_summary'])

# Preprocess for modeling
preprocessing_result = preprocess_dataset_enhanced(cleaned_df)
model_ready_df = preprocessing_result['model_ready_df']
```

### Advanced Usage with Configuration

```python
# Custom configuration for cleaning
cleaning_config = {
    "missing_threshold": 0.3,
    "outlier_detection_methods": ["iqr", "zscore", "isolation_forest"],
    "outlier_threshold": 2.5,
    "duplicate_detection": "fuzzy",
    "advanced_imputation": True,
    "data_quality_score": True
}

# Custom configuration for preprocessing
preprocessing_config = {
    "missing_threshold": 0.3,
    "variance_threshold": 0.01,
    "correlation_threshold": 0.95,
    "outlier_method": "zscore",
    "outlier_threshold": 2.5,
    "imputation_strategy": "knn",
    "scaling_method": "robust",
    "encoding_method": "auto",
    "feature_selection": True,
    "max_cardinality_onehot": 15,
    "knn_neighbors": 5
}

# Apply with custom configuration
cleaning_result = clean_dataset_enhanced(df, cleaning_config)
preprocessing_result = preprocess_dataset_enhanced(df, preprocessing_config)
```

### API Integration

The enhanced pipelines are integrated into the FastAPI backend:

```python
# Use the enhanced upload endpoint
POST /upload-enhanced
Parameters:
- file: Dataset file (CSV/Excel)
- use_enhanced: boolean (default: True)
- preprocessing_config: JSON string
- cleaning_config: JSON string
```

## Configuration Options

### Cleaning Pipeline Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `missing_threshold` | float | 0.4 | Drop columns with >40% missing values |
| `outlier_detection_methods` | list | ["iqr", "zscore", "isolation_forest"] | Methods to use for outlier detection |
| `outlier_threshold` | float | 3.0 | Threshold for Z-score outlier detection |
| `duplicate_detection` | str | "exact" | "exact" or "fuzzy" duplicate detection |
| `text_cleaning` | bool | True | Enable advanced text cleaning |
| `encoding_detection` | bool | True | Detect and fix encoding issues |
| `advanced_imputation` | bool | True | Use advanced imputation strategies |
| `data_quality_score` | bool | True | Calculate comprehensive data quality score |

### Preprocessing Pipeline Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `missing_threshold` | float | 0.4 | Drop columns with >40% missing values |
| `variance_threshold` | float | 0.01 | Remove features with variance < threshold |
| `correlation_threshold` | float | 0.95 | Remove highly correlated features |
| `outlier_method` | str | "iqr" | "iqr", "zscore", or "isolation_forest" |
| `outlier_threshold` | float | 3.0 | Threshold for outlier detection |
| `imputation_strategy` | str | "adaptive" | "adaptive", "knn", or "simple" |
| `scaling_method` | str | "standard" | "standard", "robust", or "minmax" |
| `encoding_method` | str | "auto" | "auto", "one_hot", "label", or "target" |
| `feature_selection` | bool | True | Enable automatic feature selection |
| `max_cardinality_onehot` | int | 20 | Max unique values for one-hot encoding |
| `knn_neighbors` | int | 5 | Number of neighbors for KNN imputation |

## Data Quality Score

The enhanced pipeline calculates a comprehensive data quality score with the following metrics:

- **Completeness**: Percentage of non-missing values
- **Uniqueness**: Percentage of unique rows
- **Consistency**: Data type consistency score
- **Validity**: Outlier-based validity score
- **Overall**: Weighted average of all metrics

**Quality Grades:**
- A (Excellent): 90-100
- B (Good): 80-89
- C (Fair): 70-79
- D (Poor): 60-69
- F (Very Poor): 0-59

## Enhanced Features

### 1. International Format Support

- **European decimals**: 1.234,56 → 1234.56
- **US thousands**: 1,234.56 → 1234.56
- **Scientific notation**: 1.23e+4 → 12300
- **Currency**: $1,234.56 → 1234.56
- **Percentages**: 25.5% → 25.5

### 2. Advanced Boolean Detection

Recognizes 20+ boolean patterns:
- Standard: true/false, yes/no, 1/0, y/n, t/f
- Extended: active/inactive, enabled/disabled, on/off
- Contextual: pass/fail, success/failure, approved/rejected

### 3. Smart Categorical Encoding

- **One-hot encoding**: For ≤20 unique values
- **Label encoding**: For 21-100 unique values
- **Frequency encoding**: For >100 unique values
- **Target encoding**: Available for supervised learning

### 4. Comprehensive Missing Value Analysis

- Missing value patterns and correlations
- Systematic missing value detection
- Row-level missing value analysis
- Imputation strategy recommendations

### 5. Multi-Method Outlier Detection

- **IQR Method**: Robust for non-normal distributions
- **Z-Score Method**: Good for normal distributions
- **Isolation Forest**: Multivariate outlier detection
- **Consensus Approach**: Outliers detected by multiple methods

## Performance Considerations

The enhanced pipelines include optimizations for:

- **Memory efficiency**: Minimal DataFrame copies
- **Computational efficiency**: Vectorized operations
- **Scalability**: Configurable processing depth
- **Error handling**: Graceful degradation on failures

## Testing

Run the comprehensive test suite:

```bash
python test_enhanced_pipelines.py
```

This will:
- Compare original vs enhanced pipelines
- Test special cases and edge cases
- Generate performance reports
- Validate data quality improvements

## Migration Guide

### From Original to Enhanced

1. **Import changes:**
   ```python
   # Old
   from backend.data_cleaning import clean_dataset
   from backend.preprocess import preprocess_dataset
   
   # New
   from backend.enhanced_data_cleaning import clean_dataset_enhanced
   from backend.enhanced_preprocess import preprocess_dataset_enhanced
   ```

2. **API changes:**
   ```python
   # Old
   result = clean_dataset(df)
   
   # New (with same interface)
   result = clean_dataset_enhanced(df)
   
   # New (with configuration)
   result = clean_dataset_enhanced(df, config)
   ```

3. **Response format changes:**
   - Enhanced pipelines provide additional fields:
     - `data_quality_score`: Overall quality metrics
     - `detailed_steps`: Comprehensive processing report
     - `metadata`: Pipeline configuration and statistics

## Troubleshooting

### Common Issues

1. **Memory errors with large datasets:**
   - Reduce `profiling_depth` to "basic" or "standard"
   - Disable `correlation_analysis` for very wide datasets
   - Use smaller `knn_neighbors` for KNN imputation

2. **Slow processing:**
   - Disable `advanced_imputation` if not needed
   - Use "exact" instead of "fuzzy" duplicate detection
   - Reduce `max_cardinality_onehot` for encoding

3. **Encoding issues:**
   - Ensure `encoding_detection` is enabled
   - Check file encoding before processing
   - Use UTF-8 encoding for text files

### Error Handling

The enhanced pipelines include comprehensive error handling:
- Graceful degradation on individual step failures
- Detailed error reporting in results
- Fallback to simpler methods when advanced methods fail
- Logging for debugging purposes

## Future Enhancements

Planned improvements include:
- **Anomaly detection**: Advanced pattern recognition
- **Data validation**: Custom validation rules
- **Automated EDA**: Exploratory data analysis reports
- **ML pipeline integration**: Direct model training support
- **Real-time processing**: Streaming data support

## Contributing

When contributing to the enhanced pipelines:

1. Maintain backward compatibility where possible
2. Add comprehensive tests for new features
3. Update documentation for configuration changes
4. Follow the existing code style and patterns
5. Include performance benchmarks for significant changes

## License

These enhanced pipelines are part of the AI Data Analyst Agent project and maintain the same license as the original codebase.
