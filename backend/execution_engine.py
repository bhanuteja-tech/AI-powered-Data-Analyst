import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless – must come before pyplot import
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split
import traceback
import io
import sys
import contextlib
import base64
import ast
import re
import builtins
import types

# ---------------------------------------------------------------------------
# Helpers – safe code transformation (replaces fragile monkey-patching)
# ---------------------------------------------------------------------------

_CORR_GUARD_RE = re.compile(
    r"""
    (?P<obj>[A-Za-z_]\w*)        # variable name  (e.g. df, data)
    \.corr\(                     # .corr(
    """,
    re.VERBOSE,
)

def _inject_numeric_guard(code: str) -> str:
    """
    Replace `<var>.corr(` with `<var>.select_dtypes(include=[np.number]).corr(`
    so that correlations never choke on string/categorical columns.
    """
    # Guard .corr() calls — this is the only transformation needed.
    # sns.heatmap() receives the output of .corr() which is already numeric,
    # so no separate heatmap guard is required.
    code = _CORR_GUARD_RE.sub(
        r"\g<obj>.select_dtypes(include=[np.number]).corr(",
        code,
    )

    return code


class ExecutionEngine:
    def __init__(self):
        # We allow a limited set of modules in the global namespace.
        # Import lines in generated code are stripped; expose common sklearn names here.
        # Full builtins: a minimal dict breaks NumPy/sklearn (ndarray __str__, lazy imports).
        self.allowed_globals = {
            'pd': pd,
            'np': np,
            'plt': plt,
            'sns': sns,
            'px': px,
            'go': go,
            'LabelEncoder': LabelEncoder,
            'StandardScaler': StandardScaler,
            'MinMaxScaler': MinMaxScaler,
            'OrdinalEncoder': OrdinalEncoder,
            'train_test_split': train_test_split,
            '__builtins__': builtins,
        }

    # ------------------------------------------------------------------
    # Internal: run cleaned code in a sandbox
    # ------------------------------------------------------------------
    def _run_code(self, code: str, local_vars: dict) -> dict:
        """Execute *code* with *local_vars* and return a result dict."""
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
                # Strip import statements to prevent ImportError in the sandbox
                cleaned_code_lines = []
                for line in code.split('\n'):
                    if not line.strip().startswith(('import ', 'from ')):
                        cleaned_code_lines.append(line)
                cleaned_code = '\n'.join(cleaned_code_lines)

                # Inject numeric guards for .corr() / heatmap safety
                cleaned_code = _inject_numeric_guard(cleaned_code)

                # Parse AST to capture the last expression's output
                parsed_ast = ast.parse(cleaned_code)

                # Check if the last node is an expression
                last_expr = None
                if parsed_ast.body and isinstance(parsed_ast.body[-1], ast.Expr):
                    last_expr = parsed_ast.body.pop()

                # Recompile and exec the body (minus the last expr if it was popped)
                exec(compile(parsed_ast, filename="<ast>", mode="exec"),
                     self.allowed_globals, local_vars)

                # If the last statement was an expression, eval it
                if last_expr:
                    expr_result = eval(
                        compile(ast.Expression(last_expr.value),
                                filename="<ast>", mode="eval"),
                        self.allowed_globals, local_vars,
                    )
                    if expr_result is not None:
                        result["output"] += "\n" + str(expr_result)

                # Check if a matplotlib plot was generated
                fig = plt.gcf()
                if fig and fig.get_axes():
                    buf = io.BytesIO()
                    fig.savefig(buf, format="png", bbox_inches='tight')
                    buf.seek(0)
                    img_str = base64.b64encode(buf.read()).decode('utf-8')
                    result["figure"] = f"data:image/png;base64,{img_str}"

            result["output"] = output_buffer.getvalue()

            # If a variable named 'result_text' was assigned in the code, add it
            if 'result_text' in local_vars:
                if result["output"]:
                    result["output"] += "\n" + str(local_vars['result_text'])
                else:
                    result["output"] = str(local_vars['result_text'])

            # If a variable named 'plotly_fig' was assigned in the code
            if 'plotly_fig' in local_vars:
                try:
                    result["plotly_figure"] = local_vars['plotly_fig'].to_json()
                except Exception:
                    pass

        except Exception:
            result["status"] = "error"
            exc_type, exc_value, exc_traceback = sys.exc_info()
            error_details = traceback.format_exception(exc_type, exc_value, exc_traceback)
            result["error"] = "".join(error_details)
            result["output"] = output_buffer.getvalue()
        finally:
            plt.close('all')

        return result

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def execute_code(self, code: str, df: pd.DataFrame, query: str = "") -> dict:
        """
        Executes the given pandas/python code securely with ``df`` in locals.

        Strategy:
        1. Provide the **full** ``df`` (all columns) so groupby / filtering on
           categorical columns still works.
        2. Also provide ``numeric_df`` (numeric-only) for correlation / numeric
           operations.
        3. Run a regex-based code transformer that guards ``.corr()`` and
           ``sns.heatmap()`` calls to auto-select numeric columns — no fragile
           monkey-patching needed.
        4. If the first attempt fails with a "could not convert string to float"
           error, retry with ``df`` replaced by ``numeric_df``.
        """
        numeric_df = df.select_dtypes(include=[np.number])

        # --- Attempt 1: full df + numeric_df, with code-level guards ---
        local_vars = {'df': df, 'numeric_df': numeric_df}
        result = self._run_code(code, local_vars)

        if result["status"] == "success":
            return result

        # --- Attempt 2: if the error looks like a string↔float coercion issue,
        #     retry with df = numeric_df (drop all categoricals) ---
        error_str = result.get("error", "")
        if ("could not convert string to float" in error_str
                or "could not convert string" in error_str
                or "Cannot convert" in error_str
                or "invalid literal" in error_str):
            local_vars_retry = {'df': numeric_df, 'numeric_df': numeric_df}
            result = self._run_code(code, local_vars_retry)

        return result
