"""
Execution Engine for Secure & Resilient Data Analysis
Executes LLM-generated Python/pandas/plotting code in a sandboxed environment,
captures text stdout/eval outputs, and reliably extracts Plotly and Matplotlib figures.
"""

import ast
import base64
import contextlib
import io
import math
import re
import sys
import traceback
from typing import Any, Dict, Optional

import matplotlib
matplotlib.use("Agg")  # Headless mode
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    OrdinalEncoder,
    StandardScaler,
)

# Safe builtins dictionary: explicitly excludes dangerous primitives
# (__import__, open, eval, exec, compile, globals, locals)
_SAFE_BUILTINS = {
    "abs": abs,
    "all": all,
    "any": any,
    "bin": bin,
    "bool": bool,
    "chr": chr,
    "complex": complex,
    "dict": dict,
    "divmod": divmod,
    "enumerate": enumerate,
    "filter": filter,
    "float": float,
    "format": format,
    "frozenset": frozenset,
    "hash": hash,
    "hex": hex,
    "int": int,
    "isinstance": isinstance,
    "issubclass": issubclass,
    "iter": iter,
    "len": len,
    "list": list,
    "map": map,
    "max": max,
    "min": min,
    "next": next,
    "oct": oct,
    "ord": ord,
    "pow": pow,
    "print": print,
    "range": range,
    "repr": repr,
    "reversed": reversed,
    "round": round,
    "set": set,
    "slice": slice,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "type": type,
    "zip": zip,
    "True": True,
    "False": False,
    "None": None,
    "Exception": Exception,
    "ValueError": ValueError,
    "KeyError": KeyError,
    "TypeError": TypeError,
    "IndexError": IndexError,
    "AttributeError": AttributeError,
}


def _strip_imports_ast(code: str) -> str:
    """Safely strips import statements using AST parsing without breaking multi-line syntax."""
    try:
        tree = ast.parse(code)
        filtered_body = [
            node for node in tree.body
            if not isinstance(node, (ast.Import, ast.ImportFrom))
        ]
        tree.body = filtered_body
        return ast.unparse(tree)
    except Exception:
        # Fallback to line-by-line if AST unparse fails
        cleaned = []
        for line in code.split("\n"):
            stripped = line.strip()
            if not (stripped.startswith("import ") or stripped.startswith("from ")):
                cleaned.append(line)
        return "\n".join(cleaned)


def _wrap_safe_groupby(grouped):
    """Wraps a DataFrameGroupBy object so that aggregation functions default numeric_only=True."""
    for method_name in ["mean", "median", "std", "var", "sum", "sem", "skew", "kurt"]:
        if hasattr(grouped, method_name):
            orig_method = getattr(grouped, method_name)
            def _bind(m):
                def safe_method(*args, **kwargs):
                    if "numeric_only" not in kwargs:
                        kwargs["numeric_only"] = True
                    try:
                        return m(*args, **kwargs)
                    except TypeError:
                        kwargs.pop("numeric_only", None)
                        return m(*args, **kwargs)
                return safe_method
            setattr(grouped, method_name, _bind(orig_method))
    return grouped


_PANDAS_PATCHED = False

def patch_pandas_globally():
    """Globally patches pd.DataFrame methods so aggregations, groupby, and correlations never fail on non-numeric columns."""
    global _PANDAS_PATCHED
    if _PANDAS_PATCHED:
        return
    _PANDAS_PATCHED = True

    # 1. Patch DataFrame.groupby
    orig_groupby = pd.DataFrame.groupby
    def safe_groupby(self, *args, **kwargs):
        grouped = orig_groupby(self, *args, **kwargs)
        return _wrap_safe_groupby(grouped)
    pd.DataFrame.groupby = safe_groupby

    # 2. Patch corr and cov
    for m_name in ["corr", "cov"]:
        orig_fn = getattr(pd.DataFrame, m_name)
        def _make_safe_corr_cov(fn):
            def _safe_corr_cov_method(self, *args, **kwargs):
                if "numeric_only" not in kwargs:
                    kwargs["numeric_only"] = True
                try:
                    res = fn(self, *args, **kwargs)
                    if not res.empty:
                        return res
                except Exception:
                    pass
                num_self = self.select_dtypes(include=[np.number])
                if num_self.empty:
                    return pd.DataFrame()
                kwargs["numeric_only"] = True
                return fn(num_self, *args, **kwargs)
            return _safe_corr_cov_method
        setattr(pd.DataFrame, m_name, _make_safe_corr_cov(orig_fn))

    # 3. Patch aggregations: mean, median, std, var, sum, sem, skew, kurt
    for m_name in ["mean", "median", "std", "var", "sum", "sem", "skew", "kurt"]:
        orig_agg_fn = getattr(pd.DataFrame, m_name)
        def _make_safe_agg(fn):
            def _safe_agg_method(self, *args, **kwargs):
                if "numeric_only" not in kwargs:
                    kwargs["numeric_only"] = True
                try:
                    return fn(self, *args, **kwargs)
                except Exception:
                    num_self = self.select_dtypes(include=[np.number])
                    kwargs["numeric_only"] = True
                    return fn(num_self, *args, **kwargs)
            return _safe_agg_method
        setattr(pd.DataFrame, m_name, _make_safe_agg(orig_agg_fn))


patch_pandas_globally()


def _wrap_safe_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Wraps DataFrame and ensures global safe pandas patches are active."""
    patch_pandas_globally()
    return df.copy()


class SafePandasProxy:
    """Wraps the pandas module so that calls to pd.read_csv/read_excel safely return the active DataFrame."""
    def __init__(self, target_df: pd.DataFrame):
        self._target_df = target_df

    def __getattr__(self, name: str):
        if name in ("read_csv", "read_excel", "read_table", "read_parquet", "read_feather"):
            return lambda *args, **kwargs: self._target_df
        return getattr(pd, name)


class ExecutionEngine:
    def __init__(self):
        self.allowed_globals = {
            "pd": pd,
            "np": np,
            "plt": plt,
            "sns": sns,
            "px": px,
            "go": go,
            "math": math,
            "LabelEncoder": LabelEncoder,
            "StandardScaler": StandardScaler,
            "MinMaxScaler": MinMaxScaler,
            "OrdinalEncoder": OrdinalEncoder,
            "train_test_split": train_test_split,
            "__builtins__": _SAFE_BUILTINS,
        }

    def _extract_plotly_figure(self, local_vars: Dict[str, Any]) -> Optional[str]:
        """Discovers any Plotly figure in local variables, supporting plotly_fig, fig, figure, etc."""
        fig_obj = None
        # 1. Check known variable names first
        for name in ["plotly_fig", "fig", "figure", "chart", "plot"]:
            if name in local_vars:
                val = local_vars[name]
                if isinstance(val, go.Figure) or hasattr(val, "to_json"):
                    fig_obj = val
                    break

        # 2. Search all local variables for any Figure instance
        if fig_obj is None:
            for val in local_vars.values():
                if isinstance(val, go.Figure):
                    fig_obj = val
                    break

        if fig_obj is not None:
            try:
                # Ensure responsive, prominent container sizing so it doesn't appear squished
                if hasattr(fig_obj, "update_layout"):
                    fig_obj.update_layout(
                        autosize=True,
                        height=520,
                        margin=dict(l=40, r=40, t=50, b=50),
                    )
                # Release square domain constraints on heatmaps so they stretch nicely across the chat container
                if hasattr(fig_obj, "update_yaxes"):
                    fig_obj.update_yaxes(scaleanchor=None, constrain=None)
                if hasattr(fig_obj, "update_xaxes"):
                    fig_obj.update_xaxes(constrain=None)
                return fig_obj.to_json()
            except Exception:
                try:
                    return fig_obj.to_json()
                except Exception:
                    pass

        return None

    def _run_code(self, code: str, local_vars: dict) -> dict:
        """Executes cleaned Python code within the sandbox and captures outputs."""
        output_buffer = io.StringIO()
        result = {
            "status": "success",
            "output": "",
            "error": "",
            "figure": None,
            "plotly_figure": None,
        }

        plt.clf()

        try:
            with contextlib.redirect_stdout(output_buffer), \
                 contextlib.redirect_stderr(output_buffer):

                # 1. Strip import statements cleanly
                cleaned_code = _strip_imports_ast(code)

                # 2. Strip dummy file loading (e.g. df = pd.read_csv('...')) since df is already provided
                cleaned_code = re.sub(
                    r"^\s*(?:df|data)\s*=\s*(?:pd\.)?read_(?:csv|excel|parquet|feather|table)\s*\(.*?\)\s*$",
                    "# [Dataset already provided in df]",
                    cleaned_code,
                    flags=re.MULTILINE,
                )

                # 3. Remove .show() calls that block or open browser tabs
                cleaned_code = re.sub(r"\.show\(\)", "", cleaned_code)
                cleaned_code = re.sub(r"plt\.show\(\)", "", cleaned_code)

                # 3. Parse AST to capture the last expression's return value
                parsed_ast = ast.parse(cleaned_code)

                last_expr = None
                if parsed_ast.body and isinstance(parsed_ast.body[-1], ast.Expr):
                    last_expr = parsed_ast.body.pop()

                # 4. Execute the statement body
                compiled_body = compile(parsed_ast, filename="<agent_code>", mode="exec")
                exec(compiled_body, self.allowed_globals, local_vars)

                # 5. Evaluate the trailing expression if present
                if last_expr:
                    expr_result = eval(
                        compile(ast.Expression(last_expr.value), filename="<agent_code>", mode="eval"),
                        self.allowed_globals,
                        local_vars,
                    )
                    if expr_result is not None:
                        result["output"] += "\n" + str(expr_result)

                # 6. Capture Matplotlib plot if generated (balanced size)
                fig = plt.gcf()
                if fig and fig.get_axes():
                    w, h = fig.get_size_inches()
                    if w > 10 or h > 5.5:
                        fig.set_size_inches(9.5, 5.0)
                    elif h > w * 0.7:
                        fig.set_size_inches(9.5, 5.0)
                    try:
                        fig.tight_layout()
                    except Exception:
                        pass
                    buf = io.BytesIO()
                    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
                    buf.seek(0)
                    img_str = base64.b64encode(buf.read()).decode("utf-8")
                    result["figure"] = f"data:image/png;base64,{img_str}"

            result["output"] = (output_buffer.getvalue() + result["output"]).strip()

            # Capture result_text if assigned
            if "result_text" in local_vars:
                res_txt = str(local_vars["result_text"]).strip()
                if res_txt:
                    result["output"] = f"{result['output']}\n{res_txt}".strip()

            # Capture Plotly figure
            plotly_json = self._extract_plotly_figure(local_vars)
            if plotly_json:
                result["plotly_figure"] = plotly_json

            # Always inject exact statistical metrics for correlations or plots so the insight generator has real numbers
            numeric_df = local_vars.get("numeric_df")
            df = local_vars.get("df")
            if "corr" in code.lower() and numeric_df is not None and not numeric_df.empty:
                c_mat = numeric_df.corr(numeric_only=True)
                pairs = []
                cols = c_mat.columns
                for i in range(len(cols)):
                    for j in range(i + 1, len(cols)):
                        pairs.append((cols[i], cols[j], c_mat.iloc[i, j]))
                pairs.sort(key=lambda x: abs(x[2]), reverse=True)
                top_corr_str = "\n".join([f"- {p[0]} vs {p[1]}: r = {p[2]:.2f}" for p in pairs[:5]])
                result["output"] = (result["output"] + f"\n\n[Exact Correlation Metrics]:\n{top_corr_str}").strip()
            elif not result["output"] and df is not None and not df.empty:
                num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if num_cols:
                    stats = df[num_cols[:4]].describe().round(2).to_dict()
                    result["output"] = f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns. Key metrics summary: {stats}"

        except Exception:
            result["status"] = "error"
            exc_type, exc_value, exc_traceback = sys.exc_info()
            error_details = traceback.format_exception(exc_type, exc_value, exc_traceback)
            result["error"] = "".join(error_details)
            result["output"] = output_buffer.getvalue().strip()
        finally:
            plt.close("all")

        return result

    def execute_code(self, code: str, df: pd.DataFrame, query: str = "") -> dict:
        """
        Public API: Executes code against a DataFrame with safe numeric views.
        Provides both full `df` and `numeric_df`.
        """
        safe_df = _wrap_safe_dataframe(df)
        numeric_df = safe_df.select_dtypes(include=[np.number])

        local_vars = {
            "df": safe_df,
            "numeric_df": numeric_df,
            "data": safe_df,
            "pd": SafePandasProxy(safe_df),
        }

        result = self._run_code(code, local_vars)

        # Retry once with pure numeric_df if any string-to-float or numeric conversion errors occurred
        if result["status"] == "error":
            err = result.get("error", "")
            if any(p in err.lower() for p in ["could not convert string", "cannot convert string", "invalid literal", "could not convert"]):
                local_vars_retry = {
                    "df": numeric_df,
                    "numeric_df": numeric_df,
                    "data": numeric_df,
                    "pd": SafePandasProxy(numeric_df),
                }
                retry_result = self._run_code(code, local_vars_retry)
                if retry_result.get("status") == "success":
                    result = retry_result

        return result
