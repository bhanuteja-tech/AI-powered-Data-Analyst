"""
Test script to demonstrate enhanced preprocessing and cleaning capabilities.
This script creates realistic messy datasets and shows how the enhanced pipelines handle them.
"""

import pandas as pd
import numpy as np
import json
from backend.pipelines.core.enhanced_preprocess import preprocess_dataset_enhanced
from backend.pipelines.core.enhanced_data_cleaning import clean_dataset_enhanced
from backend.pipelines.legacy.preprocess import preprocess_dataset
from backend.pipelines.legacy.data_cleaning import clean_dataset


def create_messy_dataset_1():
    """Create a realistic messy dataset with various data quality issues."""
    np.random.seed(42)
    n = 1000
    
    data = {
        # Numeric issues
        'Price': ['$1,234.56', '$2,345.67', '$3,456.78', 'N/A', '$4,567.89'] * 200,
        'Quantity': ['1,234', '2,345', '3,456', '', '4,567'] * 200,
        'Percentage': ['25.5%', '30.2%', '15.8%', 'N/A', '45.1%'] * 200,
        
        # Boolean issues
        'Active': ['TRUE', 'False', 'yes', 'NO', '1', '0', 'enabled', 'disabled'] * 125,
        'Verified': ['Y', 'N', 'true', 'false', 'T', 'F'] * 167,
        
        # Datetime issues
        'Order_Date': ['2023-01-15', '15/01/2023', 'Jan 15, 2023', '2023-01-15 14:30:00', 
                      '2023-01-16', '16/01/2023', 'Jan 16, 2023', '2023-01-16 15:45:00'] * 125,
        'Created_Time': ['2023-01-15T10:30:00Z', '2023-01-15 10:30:00', 'Jan 15 2023 10:30',
                       '2023-01-16T11:45:00Z', '2023-01-16 11:45:00', 'Jan 16 2023 11:45'] * 167,
        
        # Categorical issues
        'Category': [' Electronics ', 'electronics', 'ELECTRONICS', '  clothing  ', 'Clothing', 
                    'CLOTHING', '  books  ', 'Books', 'BOOKS'] * 112,
        'Status': ['pending', 'PENDING', 'Pending', 'completed', 'COMPLETED', 'Completed',
                  'cancelled', 'CANCELLED', 'Cancelled'] * 112,
        
        # Text issues
        'Description': ['Great product!  ', '  Excellent quality', 'Good value    ', 
                       'Poor quality!!', '  Amazing product  '] * 200,
        
        # ID columns (should be detected and dropped)
        'user_id': [f'USER_{i:06d}' for i in range(n)],
        'transaction_uuid': [f'TXN-{i:08x}' for i in range(n)],
        
        # Target variable
        'Target': [0, 1, 1, 0, 1] * 200,
        
        # Missing values
        'Missing_Num': [np.nan, 100, 200, np.nan, 300] * 200,
        'Missing_Cat': ['A', np.nan, 'B', 'C', np.nan] * 200,
        
        # Outliers
        'Normal_Num': np.random.normal(50, 10, n),
        'With_Outliers': np.concatenate([np.random.normal(50, 10, 950), [1000, -500, 2000, -1000, 1500]]),
        
        # Constant column (should be dropped)
        'Constant_Col': ['SAME_VALUE'] * n,
        
        # Mixed data types
        'Mixed_Col': ['123', '456', 'ABC', '789', 'XYZ'] * 200
    }
    
    df = pd.DataFrame(data)
    
    # Add some missing values randomly
    for col in ['Price', 'Quantity', 'Percentage', 'Active', 'Category']:
        missing_idx = np.random.choice(df.index, size=int(0.1 * len(df)), replace=False)
        df.loc[missing_idx, col] = np.nan
    
    # Add some duplicate rows
    duplicate_rows = df.sample(50).copy()
    df = pd.concat([df, duplicate_rows], ignore_index=True)
    
    return df


def create_messy_dataset_2():
    """Create a dataset with international number formats and encoding issues."""
    np.random.seed(123)
    n = 500
    
    data = {
        # European decimal format
        'European_Price': ['1.234,56', '2.345,67', '3.456,78', '4.567,89', '5.678,90'] * 100,
        
        # Scientific notation
        'Scientific_Num': ['1.23e+4', '2.34e-3', '3.45e+2', '4.56e-1', '5.67e+3'] * 100,
        
        # Mixed encoding characters
        'Text_UTF8': ['Café', 'Résumé', 'Naïve', 'Cliché', 'Fiancé'] * 100,
        
        # Phone numbers (should be treated as categorical)
        'Phone': ['(555) 123-4567', '555-123-4567', '+1 555 123 4567', '555.123.4567'] * 125,
        
        # Addresses (text)
        'Address': ['123 Main St, City, State 12345', '456 Oak Ave, Town, State 67890',
                   '789 Pine Rd, Village, State 11223'] * 167,
        
        # URLs (text)
        'Website': ['https://example.com', 'http://test.org', 'www.sample.net'] * 167,
        
        # Binary target
        'Label': ['Yes', 'No', 'Yes', 'No', 'Maybe'] * 100  # Includes a third category
    }
    
    df = pd.DataFrame(data)
    
    # Add missing values
    for col in df.columns:
        missing_idx = np.random.choice(df.index, size=int(0.15 * len(df)), replace=False)
        df.loc[missing_idx, col] = np.nan
    
    return df


def test_original_vs_enhanced_cleaning():
    """Compare original vs enhanced cleaning pipelines."""
    print("=" * 80)
    print("TESTING: Original vs Enhanced Data Cleaning")
    print("=" * 80)
    
    # Create test dataset
    df = create_messy_dataset_1()
    print(f"Original dataset shape: {df.shape}")
    print(f"Original missing values: {df.isna().sum().sum()}")
    print(f"Original dtypes:\n{df.dtypes.value_counts()}")
    print("\n" + "="*50 + "\n")
    
    # Test original cleaning
    print("Testing ORIGINAL cleaning pipeline...")
    try:
        original_result = clean_dataset(df)
        original_cleaned = original_result['cleaned_df']
        print(f"✓ Original cleaning completed")
        print(f"  Shape: {original_cleaned.shape}")
        print(f"  Missing values: {original_cleaned.isna().sum().sum()}")
        print(f"  Recommended features: {len(original_result['recommended_features'])}")
    except Exception as e:
        print(f"✗ Original cleaning failed: {e}")
        original_cleaned = df.copy()
    
    print("\n" + "="*50 + "\n")
    
    # Test enhanced cleaning
    print("Testing ENHANCED cleaning pipeline...")
    try:
        enhanced_result = clean_dataset_enhanced(df)
        enhanced_cleaned = enhanced_result['cleaned_df']
        print(f"✓ Enhanced cleaning completed")
        print(f"  Shape: {enhanced_cleaned.shape}")
        print(f"  Missing values: {enhanced_cleaned.isna().sum().sum()}")
        print(f"  Recommended features: {len(enhanced_result['recommended_features'])}")
        
        if enhanced_result.get('data_quality_score'):
            score = enhanced_result['data_quality_score']
            print(f"  Data quality score: {score['overall']:.1f}/100 ({score['grade']})")
        
        print(f"\nEnhanced cleaning summary:")
        print(enhanced_result['cleaning_summary'])
        
    except Exception as e:
        print(f"✗ Enhanced cleaning failed: {e}")
        enhanced_cleaned = df.copy()
    
    print("\n" + "="*80 + "\n")


def test_original_vs_enhanced_preprocessing():
    """Compare original vs enhanced preprocessing pipelines."""
    print("=" * 80)
    print("TESTING: Original vs Enhanced Preprocessing")
    print("=" * 80)
    
    # Create test dataset
    df = create_messy_dataset_1()
    print(f"Original dataset shape: {df.shape}")
    print("\n" + "="*50 + "\n")
    
    # Test original preprocessing
    print("Testing ORIGINAL preprocessing pipeline...")
    try:
        original_result = preprocess_dataset(df)
        original_model_ready = original_result['model_ready_df']
        print(f"✓ Original preprocessing completed")
        print(f"  Cleaned shape: {original_result['cleaned_df'].shape}")
        print(f"  Model-ready shape: {original_model_ready.shape}")
        print(f"  Recommended features: {len(original_result['recommended_features'])}")
        print(f"  Missing values in model-ready: {original_model_ready.isna().sum().sum()}")
    except Exception as e:
        print(f"✗ Original preprocessing failed: {e}")
        original_model_ready = df.copy()
    
    print("\n" + "="*50 + "\n")
    
    # Test enhanced preprocessing
    print("Testing ENHANCED preprocessing pipeline...")
    try:
        enhanced_result = preprocess_dataset_enhanced(df)
        enhanced_model_ready = enhanced_result['model_ready_df']
        print(f"✓ Enhanced preprocessing completed")
        print(f"  Cleaned shape: {enhanced_result['cleaned_df'].shape}")
        print(f"  Model-ready shape: {enhanced_model_ready.shape}")
        print(f"  Recommended features: {len(enhanced_result['recommended_features'])}")
        print(f"  Missing values in model-ready: {enhanced_model_ready.isna().sum().sum()}")
        
        print(f"\nEnhanced preprocessing summary:")
        print(enhanced_result['cleaning_summary'])
        
    except Exception as e:
        print(f"✗ Enhanced preprocessing failed: {e}")
        enhanced_model_ready = df.copy()
    
    print("\n" + "="*80 + "\n")


def test_special_cases():
    """Test special cases and edge cases."""
    print("=" * 80)
    print("TESTING: Special Cases and Edge Cases")
    print("=" * 80)
    
    # Test international formats
    print("Testing international number formats...")
    df_intl = create_messy_dataset_2()
    
    try:
        result = clean_dataset_enhanced(df_intl)
        print(f"✓ International formats handled successfully")
        print(f"  Shape: {result['cleaned_df'].shape}")
        
        # Check if European decimals were converted
        if 'European_Price' in result['cleaned_df'].columns:
            dtype = result['cleaned_df']['European_Price'].dtype
            print(f"  European_Price converted to: {dtype}")
        
    except Exception as e:
        print(f"✗ International format test failed: {e}")
    
    print("\n" + "="*50 + "\n")
    
    # Test with configuration
    print("Testing with custom configuration...")
    config = {
        "missing_threshold": 0.3,
        "outlier_method": "zscore",
        "outlier_threshold": 2.5,
        "scaling_method": "robust",
        "max_cardinality_onehot": 15
    }
    
    try:
        df_config = create_messy_dataset_1()
        result = preprocess_dataset_enhanced(df_config, config)
        print(f"✓ Custom configuration applied successfully")
        print(f"  Model-ready shape: {result['model_ready_df'].shape}")
        
        # Check if configuration was used
        report = result['metadata']['preprocessing_report']
        if report.get('scaling_applied'):
            method = report['scaling_applied']['method']
            print(f"  Scaling method used: {method}")
        
    except Exception as e:
        print(f"✗ Custom configuration test failed: {e}")
    
    print("\n" + "="*80 + "\n")


def generate_performance_report():
    """Generate a comprehensive performance report."""
    print("=" * 80)
    print("PERFORMANCE COMPARISON REPORT")
    print("=" * 80)
    
    df = create_messy_dataset_1()
    
    # Test both pipelines
    results = {}
    
    # Original pipelines
    try:
        import time
        start = time.time()
        orig_clean = clean_dataset(df)
        orig_clean_time = time.time() - start
        
        start = time.time()
        orig_prep = preprocess_dataset(df)
        orig_prep_time = time.time() - start
        
        results['original'] = {
            'cleaning_time': orig_clean_time,
            'preprocessing_time': orig_prep_time,
            'cleaned_shape': orig_clean['cleaned_df'].shape,
            'model_ready_shape': orig_prep['model_ready_df'].shape,
            'features_recommended': len(orig_prep['recommended_features'])
        }
    except Exception as e:
        results['original'] = {'error': str(e)}
    
    # Enhanced pipelines
    try:
        start = time.time()
        enh_clean = clean_dataset_enhanced(df)
        enh_clean_time = time.time() - start
        
        start = time.time()
        enh_prep = preprocess_dataset_enhanced(df)
        enh_prep_time = time.time() - start
        
        results['enhanced'] = {
            'cleaning_time': enh_clean_time,
            'preprocessing_time': enh_prep_time,
            'cleaned_shape': enh_clean['cleaned_df'].shape,
            'model_ready_shape': enh_prep['model_ready_df'].shape,
            'features_recommended': len(enh_prep['recommended_features']),
            'data_quality_score': enh_clean.get('data_quality_score', {}).get('overall', 0)
        }
    except Exception as e:
        results['enhanced'] = {'error': str(e)}
    
    # Print comparison
    print("ORIGINAL PIPELINE:")
    if 'error' not in results['original']:
        orig = results['original']
        print(f"  Cleaning time: {orig['cleaning_time']:.3f}s")
        print(f"  Preprocessing time: {orig['preprocessing_time']:.3f}s")
        print(f"  Total time: {orig['cleaning_time'] + orig['preprocessing_time']:.3f}s")
        print(f"  Cleaned shape: {orig['cleaned_shape']}")
        print(f"  Model-ready shape: {orig['model_ready_shape']}")
        print(f"  Features recommended: {orig['features_recommended']}")
    else:
        print(f"  Error: {results['original']['error']}")
    
    print("\nENHANCED PIPELINE:")
    if 'error' not in results['enhanced']:
        enh = results['enhanced']
        print(f"  Cleaning time: {enh['cleaning_time']:.3f}s")
        print(f"  Preprocessing time: {enh['preprocessing_time']:.3f}s")
        print(f"  Total time: {enh['cleaning_time'] + enh['preprocessing_time']:.3f}s")
        print(f"  Cleaned shape: {enh['cleaned_shape']}")
        print(f"  Model-ready shape: {enh['model_ready_shape']}")
        print(f"  Features recommended: {enh['features_recommended']}")
        print(f"  Data quality score: {enh['data_quality_score']:.1f}/100")
    else:
        print(f"  Error: {results['enhanced']['error']}")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    print("🚀 Testing Enhanced Data Processing Pipelines")
    print("This script demonstrates the improvements over the original implementation.\n")
    
    # Run all tests
    test_original_vs_enhanced_cleaning()
    test_original_vs_enhanced_preprocessing()
    test_special_cases()
    generate_performance_report()
    
    print("✅ All tests completed!")
    print("\nThe enhanced pipelines provide:")
    print("  • Better data type detection and conversion")
    print("  • Advanced missing value analysis and imputation")
    print("  • Multiple outlier detection methods")
    print("  • Comprehensive data quality scoring")
    print("  • Enhanced categorical encoding strategies")
    print("  • International format support")
    print("  • Detailed reporting and recommendations")
    print("  • Configurable processing parameters")
