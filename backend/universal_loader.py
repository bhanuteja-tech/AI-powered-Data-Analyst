"""
Universal Dataset Loader & Schema Profiler
Handles diverse formats (CSV, TSV, Excel, JSON, Parquet), encodings, delimiters,
malformed headers, and memory optimization with intelligent in-memory caching.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import chardet

logger = logging.getLogger(__name__)

# In-memory cache: (file_path, mtime) -> (df, metadata)
_LOADER_CACHE: Dict[Tuple[str, float], Tuple[pd.DataFrame, Dict[str, Any]]] = {}


def _detect_encoding(file_path: str, default: str = "utf-8") -> str:
    """Detects character encoding using chardet on the first 32KB of the file."""
    try:
        with open(file_path, "rb") as f:
            raw = f.read(32768)
        if not raw:
            return default
        detected = chardet.detect(raw)
        encoding = detected.get("encoding")
        confidence = detected.get("confidence", 0.0)
        if encoding and confidence >= 0.7:
            # Normalize common aliases
            enc_clean = encoding.lower()
            if "ascii" in enc_clean or "utf-8" in enc_clean:
                return "utf-8"
            return encoding
        return default
    except Exception as e:
        logger.warning(f"Encoding detection failed for {file_path}: {e}")
        return default


def _detect_delimiter(file_path: str, encoding: str = "utf-8") -> str:
    """Detects CSV delimiter (comma, semicolon, tab, pipe) reliably."""
    candidates = [",", ";", "\t", "|"]
    try:
        sample_lines = []
        with open(file_path, "r", encoding=encoding, errors="replace") as f:
            for _ in range(25):
                line = f.readline()
                if not line:
                    break
                # Skip comments or empty lines
                stripped = line.strip()
                if stripped and not stripped.startswith(("#", "//", "/*")):
                    sample_lines.append(stripped)

        if not sample_lines:
            return ","

        # Count candidate frequencies per line
        counts = {c: [line.count(c) for line in sample_lines] for c in candidates}
        # A good delimiter has a consistent positive count across all lines
        best_delim = ","
        best_score = -1
        for delim, line_counts in counts.items():
            if all(cnt > 0 for cnt in line_counts):
                # Variance penalty: lower variance is better
                variance = float(np.var(line_counts)) if len(line_counts) > 1 else 0.0
                mean_cnt = float(np.mean(line_counts))
                score = mean_cnt / (1.0 + variance)
                if score > best_score:
                    best_score = score
                    best_delim = delim

        return best_delim
    except Exception as e:
        logger.warning(f"Delimiter detection failed for {file_path}: {e}")
        return ","


def optimize_dataframe_memory(df: pd.DataFrame) -> pd.DataFrame:
    """Downcasts float and integer columns to minimize memory footprint without data loss."""
    try:
        for col in df.columns:
            col_type = df[col].dtype
            if pd.api.types.is_integer_dtype(col_type):
                df[col] = pd.to_numeric(df[col], downcast="integer")
            elif pd.api.types.is_float_dtype(col_type):
                df[col] = pd.to_numeric(df[col], downcast="float")
    except Exception as e:
        logger.debug(f"Memory optimization skipped on some columns: {e}")
    return df


def sanitize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans column headers: strips whitespace, removes newlines, and handles unnamed index columns."""
    new_cols = []
    seen = {}
    drop_cols = []

    for idx, col in enumerate(df.columns):
        col_str = str(col).strip()
        col_str = re.sub(r"[\r\n]+", " ", col_str)
        # Check for un-named index columns like "Unnamed: 0"
        if re.match(r"^unnamed:\s*\d+$", col_str.lower()):
            # If values match row indices, mark for dropping
            series = df[col]
            if series.nunique() == len(df) and pd.api.types.is_numeric_dtype(series):
                drop_cols.append(col)
                continue

        # Handle duplicate column names gracefully
        if col_str in seen:
            seen[col_str] += 1
            col_str = f"{col_str}_{seen[col_str]}"
        else:
            seen[col_str] = 0

        new_cols.append(col_str)

    if drop_cols:
        df = df.drop(columns=drop_cols)

    df.columns = [str(c).strip() for c in new_cols]
    return df


def load_dataset(
    file_path: str,
    optimize_memory: bool = True,
    use_cache: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Universal dataset reader. Supports CSV, TSV, Excel, JSON, Parquet, and text files.
    Recovers from encoding mismatches, malformed headers, and ragged rows.
    """
    path_obj = Path(file_path).resolve()
    if not path_obj.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    mtime = path_obj.stat().st_mtime
    cache_key = (str(path_obj), mtime)

    if use_cache and cache_key in _LOADER_CACHE:
        cached_df, cached_meta = _LOADER_CACHE[cache_key]
        return cached_df.copy(), dict(cached_meta)

    ext = path_obj.suffix.lower()
    df: Optional[pd.DataFrame] = None
    meta: Dict[str, Any] = {
        "file_name": path_obj.name,
        "format": ext.lstrip("."),
        "file_size_bytes": path_obj.stat().st_size,
    }

    # 1. Parquet
    if ext in [".parquet", ".pq"]:
        try:
            df = pd.read_parquet(str(path_obj), engine="pyarrow")
            meta["engine"] = "pyarrow"
        except Exception as e:
            df = pd.read_parquet(str(path_obj))
            meta["engine"] = "fastparquet"

    # 2. Excel
    elif ext in [".xlsx", ".xls", ".xlsm", ".xlsb"]:
        try:
            df = pd.read_excel(str(path_obj), engine="openpyxl")
            meta["engine"] = "openpyxl"
        except Exception:
            df = pd.read_excel(str(path_obj))
            meta["engine"] = "auto"

    # 3. JSON / JSON Lines
    elif ext in [".json", ".jsonl", ".ndjson"]:
        try:
            # Try JSON Lines first
            df = pd.read_json(str(path_obj), lines=True)
            meta["engine"] = "json_lines"
        except Exception:
            try:
                df = pd.read_json(str(path_obj))
                meta["engine"] = "json"
            except Exception:
                # Handle nested dict / list records
                with open(str(path_obj), "r", encoding="utf-8", errors="replace") as jf:
                    data = json.load(jf)
                df = pd.json_normalize(data)
                meta["engine"] = "json_normalize"

    # 4. Delimited Text (CSV, TSV, TXT, DAT, etc.)
    else:
        encoding = _detect_encoding(str(path_obj))
        delimiter = _detect_delimiter(str(path_obj), encoding=encoding)
        meta["detected_encoding"] = encoding
        meta["detected_delimiter"] = delimiter

        # Reading attempts with fallback strategies
        attempts = [
            # Fast C engine with detected params
            {"sep": delimiter, "encoding": encoding, "engine": "c", "on_bad_lines": "skip"},
            # Python engine (handles regex / complex delimiters)
            {"sep": delimiter, "encoding": encoding, "engine": "python", "on_bad_lines": "skip"},
            # Latin-1 fallback
            {"sep": delimiter, "encoding": "latin1", "engine": "c", "on_bad_lines": "skip"},
            # Comma fallback
            {"sep": ",", "encoding": "utf-8", "engine": "python", "on_bad_lines": "skip"},
        ]

        last_error = None
        for opts in attempts:
            try:
                df = pd.read_csv(str(path_obj), **opts)
                if df is not None and not df.empty:
                    meta["applied_options"] = opts
                    break
            except Exception as e:
                last_error = e
                continue

        if df is None:
            raise ValueError(f"Unable to parse dataset {path_obj.name}: {last_error}")

    # Sanitize columns
    df = sanitize_column_names(df)

    # Memory optimization
    if optimize_memory:
        orig_mem = df.memory_usage(deep=True).sum()
        df = optimize_dataframe_memory(df)
        new_mem = df.memory_usage(deep=True).sum()
        meta["memory_reduced_bytes"] = int(orig_mem - new_mem)

    meta["rows"] = len(df)
    meta["columns"] = len(df.columns)
    meta["column_names"] = df.columns.tolist()

    if use_cache:
        _LOADER_CACHE[cache_key] = (df.copy(), dict(meta))

    return df, meta


def get_dataset_profiling_snippet(df: pd.DataFrame, max_rows: int = 3) -> str:
    """Generates a rich, token-efficient text summary of the dataset for LLM prompts."""
    buffer = []
    buffer.append(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    buffer.append("\nColumns & Data Types:")
    for col, dtype in df.dtypes.items():
        null_count = int(df[col].isnull().sum())
        null_pct = round((null_count / max(len(df), 1)) * 100, 1)
        nunique = int(df[col].nunique(dropna=True))
        buffer.append(f"- `{col}` ({dtype}): {nunique} unique values, {null_pct}% missing")

    buffer.append("\nNumeric Summary (Key Stats):")
    numeric_df = df.select_dtypes(include=[np.number])
    if not numeric_df.empty:
        stats = numeric_df.describe().T[["mean", "std", "min", "50%", "max"]].round(2)
        buffer.append(stats.to_string())
    else:
        buffer.append("(No numeric columns found)")

    buffer.append(f"\nSample Data (First {min(max_rows, len(df))} rows):")
    buffer.append(df.head(max_rows).to_string())

    return "\n".join(buffer)


def clear_loader_cache() -> None:
    """Clears in-memory dataset cache."""
    _LOADER_CACHE.clear()
