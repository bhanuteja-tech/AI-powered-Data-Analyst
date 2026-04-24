import pandas as pd
import numpy as np
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
import builtins

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

    def execute_code(self, code: str, df: pd.DataFrame) -> dict:
        """
        Executes the given pandas/python code securely with df in locals.
        Returns a dictionary with status, output text, error, and base64 encoded plotting result if any.
        """
        local_vars = {'df': df}
        output_buffer = io.StringIO()
        result = {
            "status": "success",
            "output": "",
            "error": "",
            "figure": None,
            "plotly_figure": None
        }

        # Clear any existing plots to prevent interference
        plt.clf()

        try:
            with contextlib.redirect_stdout(output_buffer), contextlib.redirect_stderr(output_buffer):
                # Strip import statements to prevent ImportError in the sandbox
                cleaned_code_lines = []
                for line in code.split('\n'):
                    if not line.strip().startswith(('import ', 'from ')):
                        cleaned_code_lines.append(line)
                cleaned_code = '\n'.join(cleaned_code_lines)

                # Parse AST to capture the last expression's output
                parsed_ast = ast.parse(cleaned_code)
                
                # Check if the last node is an expression
                last_expr = None
                if parsed_ast.body and isinstance(parsed_ast.body[-1], ast.Expr):
                    last_expr = parsed_ast.body.pop()

                # Recompile and exec the body (minus the last expr if it was popped)
                exec(compile(parsed_ast, filename="<ast>", mode="exec"), self.allowed_globals, local_vars)

                # If the last statement was an expression, eval it to get the direct output
                if last_expr:
                    expr_result = eval(compile(ast.Expression(last_expr.value), filename="<ast>", mode="eval"), self.allowed_globals, local_vars)
                    if expr_result is not None:
                        result["output"] += "\n" + str(expr_result)

                # Check if a plot was generated
                fig = plt.gcf()
                if fig and fig.get_axes():
                    # Save the figure to a bytes buffer
                    buf = io.BytesIO()
                    fig.savefig(buf, format="png", bbox_inches='tight')
                    buf.seek(0)
                    # Encode to base64
                    img_str = base64.b64encode(buf.read()).decode('utf-8')
                    result["figure"] = f"data:image/png;base64,{img_str}"

            result["output"] = output_buffer.getvalue()
            
            # If a variable named 'result_text' was assigned in the code, add it to output
            if 'result_text' in local_vars:
                if result["output"]:
                    result["output"] += "\n" + str(local_vars['result_text'])
                else:
                    result["output"] = str(local_vars['result_text'])

            # If a variable named 'plotly_fig' was assigned in the code
            if 'plotly_fig' in local_vars:
                try:
                    result["plotly_figure"] = local_vars['plotly_fig'].to_json()
                except Exception as ex:
                    pass

        except Exception as e:
            result["status"] = "error"
            exc_type, exc_value, exc_traceback = sys.exc_info()
            error_details = traceback.format_exception(exc_type, exc_value, exc_traceback)
            result["error"] = "".join(error_details)
            result["output"] = output_buffer.getvalue()
        finally:
            # Clear plot again after capturing
            plt.close('all')

        return result
