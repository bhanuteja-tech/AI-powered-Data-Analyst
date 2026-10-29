"""
Tests for Universal Dataset Loader.
Verifies multi-format ingestion, encoding detection, delimiter detection, and memory optimization.
"""

import os
import tempfile
import pandas as pd
import numpy as np
import pytest
from backend.universal_loader import (
    load_dataset,
    optimize_dataframe_memory,
    sanitize_column_names,
    get_dataset_profiling_snippet,
)


def test_standard_csv_loading():
    with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False, encoding="utf-8") as f:
        f.write("id,name,value\n1,Alice,10.5\n2,Bob,20.0\n")
        f_path = f.name

    try:
        df, meta = load_dataset(f_path, use_cache=False)
        assert df.shape == (2, 3)
        assert list(df.columns) == ["id", "name", "value"]
        assert meta["rows"] == 2
    finally:
        os.remove(f_path)


def test_semicolon_delimited_csv():
    with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False, encoding="utf-8") as f:
        f.write("product;price;rating\nWidget;9.99;4.5\nGadget;19.99;4.8\n")
        f_path = f.name

    try:
        df, meta = load_dataset(f_path, use_cache=False)
        assert df.shape == (2, 3)
        assert "price" in df.columns
        assert meta.get("detected_delimiter") == ";"
    finally:
        os.remove(f_path)


def test_tab_delimited_tsv():
    with tempfile.NamedTemporaryFile(suffix=".tsv", mode="w", delete=False, encoding="utf-8") as f:
        f.write("col_a\tcol_b\n100\t200\n300\t400\n")
        f_path = f.name

    try:
        df, meta = load_dataset(f_path, use_cache=False)
        assert df.shape == (2, 2)
        assert "col_a" in df.columns
    finally:
        os.remove(f_path)


def test_json_lines_loading():
    with tempfile.NamedTemporaryFile(suffix=".jsonl", mode="w", delete=False, encoding="utf-8") as f:
        f.write('{"user": "u1", "score": 95}\n{"user": "u2", "score": 88}\n')
        f_path = f.name

    try:
        df, meta = load_dataset(f_path, use_cache=False)
        assert df.shape == (2, 2)
        assert "user" in df.columns
    finally:
        os.remove(f_path)


def test_parquet_loading():
    df_orig = pd.DataFrame({"num": [1, 2, 3], "label": ["x", "y", "z"]})
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        f_path = f.name

    try:
        df_orig.to_parquet(f_path, index=False)
        df, meta = load_dataset(f_path, use_cache=False)
        assert df.shape == (3, 2)
        assert list(df.columns) == ["num", "label"]
    finally:
        os.remove(f_path)


def test_latin1_encoding_recovery():
    with tempfile.NamedTemporaryFile(suffix=".csv", mode="wb", delete=False) as f:
        # Write bytes with Latin-1 specific characters (e.g. accented é = 0xE9)
        f.write("id,city\n1,Montr\xe9al\n2,Z\xfcrich\n".encode("latin1"))
        f_path = f.name

    try:
        df, meta = load_dataset(f_path, use_cache=False)
        assert df.shape == (2, 2)
        assert "city" in df.columns
    finally:
        os.remove(f_path)


def test_memory_downcasting():
    df = pd.DataFrame({
        "large_int": np.array([1, 2, 3], dtype=np.int64),
        "large_float": np.array([1.5, 2.5, 3.5], dtype=np.float64)
    })
    optimized = optimize_dataframe_memory(df)
    assert optimized["large_int"].dtype == np.int8
    assert optimized["large_float"].dtype == np.float32


def test_sanitize_column_names():
    df = pd.DataFrame({
        "  Trimmed Col  ": [1, 2],
        "Multi\nLine": [3, 4],
        "Duplicate": [5, 6],
    })
    df.columns = ["  Trimmed Col  ", "Multi\nLine", "Duplicate"]
    sanitized = sanitize_column_names(df)
    assert "Trimmed Col" in sanitized.columns
    assert "Multi Line" in sanitized.columns


def test_dataset_profiling_snippet():
    df = pd.DataFrame({
        "age": [25, 30, 35, 40],
        "salary": [50000.0, 60000.0, 75000.0, 80000.0],
        "dept": ["HR", "IT", "IT", "Finance"]
    })
    snippet = get_dataset_profiling_snippet(df)
    assert "Shape: 4 rows, 3 columns" in snippet
    assert "age" in snippet
    assert "salary" in snippet
