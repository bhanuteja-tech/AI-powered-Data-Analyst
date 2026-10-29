"""
Tests for Execution Engine sandbox security and plotting extraction.
"""

import pandas as pd
import pytest
from backend.execution_engine import ExecutionEngine


def test_basic_pandas_execution():
    ee = ExecutionEngine()
    df = pd.DataFrame({"a": [10, 20, 30], "b": [1, 2, 3]})
    res = ee.execute_code("print(df['a'].mean())", df)
    assert res["status"] == "success"
    assert "20.0" in res["output"]


def test_safe_correlation_with_categorical_columns():
    """Verifies that .corr() does not fail on mixed DataFrames containing string/text columns."""
    ee = ExecutionEngine()
    df = pd.DataFrame({
        "num1": [1, 2, 3, 4],
        "num2": [2, 4, 6, 8],
        "category": ["A", "B", "A", "B"],
    })
    res = ee.execute_code("print(df.corr())", df)
    assert res["status"] == "success"
    assert "num1" in res["output"]


def test_plotly_figure_capture_fig():
    """Verifies that assign to 'fig' captures plotly_figure properly."""
    ee = ExecutionEngine()
    df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
    code = "fig = px.bar(df, x='x', y='y')"
    res = ee.execute_code(code, df)
    assert res["status"] == "success"
    assert res["plotly_figure"] is not None


def test_plotly_figure_capture_plotly_fig():
    """Verifies that assign to 'plotly_fig' also captures plotly_figure."""
    ee = ExecutionEngine()
    df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
    code = "plotly_fig = px.scatter(df, x='x', y='y')"
    res = ee.execute_code(code, df)
    assert res["status"] == "success"
    assert res["plotly_figure"] is not None


def test_sandbox_disallows_dangerous_builtins():
    """Verifies that __import__ is not accessible in execution scope."""
    ee = ExecutionEngine()
    df = pd.DataFrame({"a": [1]})
    res = ee.execute_code("__import__('os').system('echo hack')", df)
    assert res["status"] == "error"
    assert "not defined" in res["error"] or "NameError" in res["error"]


def test_safe_groupby_mean_with_mixed_string_columns():
    """Verifies df.groupby().mean() does not fail on mixed datasets with string columns."""
    ee = ExecutionEngine()
    df = pd.DataFrame({
        "sex": ["male", "female", "male", "female"],
        "survived": [0, 1, 0, 1],
        "fare": [7.25, 71.28, 8.05, 53.10],
        "town": ["Southampton", "Cherbourg", "Southampton", "Southampton"],
    })
    code = """
import plotly.express as px
grouped = df.groupby('sex').mean().reset_index()
fig = px.bar(grouped, x='sex', y='survived')
"""
    res = ee.execute_code(code, df)
    assert res["status"] == "success", f"Failed with: {res.get('error')}"
    assert res["plotly_figure"] is not None


def test_categorical_column_bar_chart_does_not_drop_sex():
    """Verifies px.bar on categorical column 'sex' does not get overwritten by numeric_df."""
    ee = ExecutionEngine()
    df = pd.DataFrame({
        "survived": [0, 1, 0, 1],
        "pclass": [3, 1, 3, 1],
        "sex": ["male", "female", "male", "female"],
        "age": [22.0, 38.0, 26.0, 35.0],
        "fare": [7.25, 71.28, 7.92, 53.10],
    })
    code = """
import plotly.express as px
counts = df['sex'].value_counts().reset_index()
counts.columns = ['sex', 'count']
fig = px.bar(counts, x='sex', y='count')
"""
    res = ee.execute_code(code, df)
    assert res["status"] == "success", f"Failed with: {res.get('error')}"
    assert res["plotly_figure"] is not None

