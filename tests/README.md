# Test Suite

This directory contains the test suite for the AI Data Analyst Agent.

## Test Files

### `test_enhanced_pipelines.py`
Comprehensive test suite that validates the enhanced data processing pipelines against the original implementation.

**Test Coverage:**
- Data type conversion accuracy
- Missing value handling strategies
- Outlier detection methods
- Categorical encoding approaches
- Data quality scoring
- Performance benchmarks
- Error handling and edge cases

## Running Tests

### Method 1: Run Directly
```bash
cd ai-data-analyst-agent
python tests/test_enhanced_pipelines.py
```

### Method 2: Using pytest (Recommended)
```bash
cd ai-data-analyst-agent
python -m pytest tests/ -v
```

### Method 3: With Coverage Report
```bash
cd ai-data-analyst-agent
python -m pytest tests/ --cov=backend --cov-report=html
```

## Test Results

The test suite will:
1. **Create realistic messy datasets** with various data quality issues
2. **Compare original vs enhanced pipelines** on the same data
3. **Generate performance reports** showing improvements
4. **Validate data quality scoring** accuracy
5. **Test edge cases** and error handling

## Expected Output

When running the tests, you should see:
- ✅ Successful imports of both pipeline versions
- 📊 Performance comparisons between versions
- 🎯 Data quality improvements demonstrated
- ⚡ Processing time benchmarks
- 📈 Feature engineering enhancements

## Troubleshooting

### Import Errors
If you get import errors, ensure:
1. You're in the correct directory (`ai-data-analyst-agent/`)
2. All dependencies are installed (`pip install -r requirements.txt`)
3. Python path includes the project root

### Missing Dependencies
```bash
pip install -r requirements.txt
```

### Permission Issues
```bash
# On Windows, run as administrator if needed
# On Linux/Mac, check file permissions
chmod +x tests/*.py
```

## Adding New Tests

When adding new test files:
1. Follow the naming convention `test_*.py`
2. Import from the organized structure:
   ```python
   from backend.pipelines.core import enhanced_functions
   from backend.pipelines.legacy import original_functions
   ```
3. Include comprehensive assertions
4. Add documentation for test purpose
5. Update this README file

## Test Data

Tests generate synthetic data internally, so no external test files are needed. This ensures:
- Consistent test results
- No data privacy concerns
- Fast test execution
- Reproducible testing

## Continuous Integration

These tests are designed to run in CI/CD pipelines:
- ✅ No external dependencies
- ✅ Fast execution (< 2 minutes)
- ✅ Clear pass/fail output
- ✅ Detailed error reporting
