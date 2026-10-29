# Codebase Audit & Problem Analysis Report

## Executive Summary
This document provides a comprehensive analysis of the **AI Data Analyst Agent** codebase, explains the root cause of the unwanted plot generation shown in the screenshot, identifies critical architectural, security, execution, pipeline, and UI flaws, and provides actionable, step-by-step solutions to address each issue.

---

## 1. Deep Dive: Screenshot Issue & Guardrail Failure Analysis

### 1.1 The User's Scenario
* **User Query:** `"what type of questions that i can ask on this dataset (plotting questions)?"`
* **Model Used:** `qwen2.5-coder:3b` via Ollama (`http://localhost:11434/v1`)
* **Expected Result:** A list of recommended/suggested plotting questions or analytical ideas tailored to the dataset columns **without** executing any code or rendering any chart.
* **Actual Observed Result:**
  1. The system took **21.53 seconds** to respond.
  2. The LLM generated a text list of 10 questions.
  3. **A Plotly correlation heatmap was executed and rendered below the answer**, despite the user never asking for a heatmap or any plot to be rendered.
  4. The analysis code (`px.imshow(numeric_df.corr(), ...)`) was executed on the dataset.

---

### 1.2 Root Cause Analysis: Why Did This Happen?

Tracing the exact execution flow through [workflow.py](file:///c:/Users/bhanu/Downloads/Data-Analyst-Agent-20260411T135148Z-3-001/Data-Analyst-Agent/ai-data-analyst-agent/backend/agents/workflow.py) and [streamlit_app.py](file:///c:/Users/bhanu/Downloads/Data-Analyst-Agent-20260411T135148Z-3-001/Data-Analyst-Agent/ai-data-analyst-agent/frontend/streamlit_app.py):

#### Reason 1: The LangGraph Initial Router Disregards Meta/Conversational Queries in Analysis Mode
In [workflow.py:406-412](file:///c:/Users/bhanu/Downloads/Data-Analyst-Agent-20260411T135148Z-3-001/Data-Analyst-Agent/ai-data-analyst-agent/backend/agents/workflow.py#L406-L412):
```python
def route_query_initial(state: AgentState):
    if state.get("query", "").strip() == "" and state.get("chat_mode") != "chat":
        return "generate_suggestions"
    if state.get("chat_mode") == "chat":
        return "conversational_response"
    return "understand_query"
```
* The default mode in the Streamlit UI is `"📊 Data Analysis (Code & Charts)"` (`chat_mode = "analysis"`).
* The user typed a question into the chat input. Because `query` was not empty (`"what type of questions..."`) and `chat_mode != "chat"`, the query was routed to `understand_query`.
* The system assumed that **any non-empty query in analysis mode must be an execution request**.

#### Reason 2: The Intent Classifier Forces a Tri-State Decision With No Option for "Meta/Question Suggestions"
In [workflow.py:193-205](file:///c:/Users/bhanu/Downloads/Data-Analyst-Agent-20260411T135148Z-3-001/Data-Analyst-Agent/ai-data-analyst-agent/backend/agents/workflow.py#L193-L205):
```text
Determine the user's INTENT based on the current prompt and the chat history.

OPTIONS:
1. "VISUALIZATION": User explicitly asks for a graph, plot, chart, heatmap, scatter, bar, line, histogram, box plot, or any data visualization.
2. "DATA_ONLY": User asks a quantitative question requiring pandas code to calculate (e.g., max, min, average, sum, count, correlation) but no chart.
3. "OFF_TOPIC": User asks for something completely unrelated to data analysis (e.g., "write a poem", "what's the weather").

IMPORTANT: Be generous in classifying as VISUALIZATION or DATA_ONLY. If the user mentions data, columns, statistics, or comparisons related to the dataset, it is NOT off-topic.

Respond with exactly ONE word: VISUALIZATION or DATA_ONLY or OFF_TOPIC.
```
* The prompt only gave the LLM three buckets: `VISUALIZATION`, `DATA_ONLY`, or `OFF_TOPIC`.
* Asking *"what type of questions that i can ask on this dataset (plotting questions)?"* is clearly about the dataset, so the prompt explicitly forbade classifying it as `OFF_TOPIC` (*"Be generous... If user mentions data... it is NOT off-topic"*).
* Because the user query explicitly included the word **"plotting"**, the LLM selected **`VISUALIZATION`**.
* **The system had no intent category for `META_EXPLORATION` or `SUGGESTION_REQUEST` or `GENERAL_INQUIRY`.**

#### Reason 3: Defective Underspecified Guardrail
In [workflow.py:151-182](file:///c:/Users/bhanu/Downloads/Data-Analyst-Agent-20260411T135148Z-3-001/Data-Analyst-Agent/ai-data-analyst-agent/backend/agents/workflow.py#L151-L182):
```python
viz_markers = ["plot", "chart", "graph", "visual", "visualize", "heatmap", "scatter", "bar", "line", "hist", "box", "violin", "facet", "facetgrid", "facet grid", "pairplot"]
if any(m in q for m in viz_markers):
    cols = state.get("columns", []) or []
    mentions_col = any(c.lower() in q for c in cols if isinstance(c, str) and c)
    wants_facet = ("facet" in q) or ("facetgrid" in q) or ("facet grid" in q)
    if wants_facet and not mentions_col:
        ... # requests clarification
```
* Notice the bug: The comment says *"Strict mode: underspecified visualization requests -> ask clarification, do not execute"*, but the code **only** checks `if wants_facet and not mentions_col`.
* If a query matches `viz_markers` (like "plot" in "plotting") but does NOT contain "facet", the guardrail **does nothing**! It lets the query fall through to code generation!

#### Reason 4: Forced Code Generation on an Abstract Meta-Question
In [workflow.py:423-428](file:///c:/Users/bhanu/Downloads/Data-Analyst-Agent-20260411T135148Z-3-001/Data-Analyst-Agent/ai-data-analyst-agent/backend/agents/workflow.py#L423-L428):
```python
def route_after_understand(state: AgentState):
    if state.get("off_topic"):
        return "reject_off_topic"
    if state.get("needs_clarification"):
        return "request_clarification"
    return "generate_code"
```
Because `off_topic` is `False` and `needs_clarification` is `False`, the graph routed directly to `generate_code`.
The prompt instructed the code model (`qwen2.5-coder:3b`):
> *"Your task is to write valid Python code to answer the user's query."*
> *"USER QUERY: what type of questions that i can ask on this dataset (plotting questions)?"*
> *"NEEDS VISUALIZATION: True"*

Because the model was commanded to write Python code for a visualization, it wrote code to compute correlations and display a Plotly heatmap (`px.imshow(numeric_df.corr())`), while printing questions to stdout!

#### Reason 5: Execution Engine Executes and Frontend Renders Unconditionally
1. `execute_code_node` ran the generated script.
2. The script created a Plotly figure `plotly_fig`.
3. `execution_result` captured `plotly_figure`.
4. In [streamlit_app.py:731-736](file:///c:/Users/bhanu/Downloads/Data-Analyst-Agent-20260411T135148Z-3-001/Data-Analyst-Agent/ai-data-analyst-agent/frontend/streamlit_app.py#L731-L736), the frontend checks:
   ```python
   if plotly_fig:
       fig = pio.from_json(plotly_fig)
       st.plotly_chart(fig, width='stretch')
   ```
   Because `plotly_fig` was present in the API response, Streamlit drew the chart!

#### Reason 6: Why Did It Take 21.5 Seconds?
In [workflow.py:39-56](file:///c:/Users/bhanu/Downloads/Data-Analyst-Agent-20260411T135148Z-3-001/Data-Analyst-Agent/ai-data-analyst-agent/backend/agents/workflow.py#L39-L56):
`get_llm()` executes a synchronous HTTP check to `requests.get(f"{_check_url}/api/tags", timeout=5)` **every single time `get_llm()` is called**.
In a single turn, `get_llm()` is invoked by:
1. `understand_query_node`
2. `generate_code_node`
3. `generate_insights_node`

This creates up to **15 seconds** of redundant network handshake delays on top of the inference time!

---

## 2. Comprehensive Inventory of Problems in the Codebase

Beyond the immediate plotting issue, the codebase has several critical flaws across Security, Multi-Agent Architecture, Sandboxing, Data Pipelines, API Design, and Frontend UX.

### Problem Category A: Agent Workflow & Guardrail Deficiencies
1. **Lack of Fine-Grained Intent Classification:**
   - Only 3 intents exist: `VISUALIZATION`, `DATA_ONLY`, `OFF_TOPIC`.
   - Missing intents: `META_QUESTION` (asking what questions/plots are possible), `SCHEMA_INQUIRY` (asking about column types, data shape), `GENERAL_DATA_CHAT` (interpreting existing insights or general reasoning without code), and `AMBIGUOUS_REQUEST`.
2. **Artificial "Mode" Schism (Analysis Mode vs Chat Mode):**
   - Users are forced to manually switch between `"📊 Data Analysis"` and `"💬 Chat"`.
   - If in Analysis mode, any natural language question triggers full code generation and execution.
   - If in Chat mode, asking *"Plot a histogram of ages"* does not generate a plot because code execution is disabled in Chat mode.
   - The agent should be **unified and autonomous**: it should determine whether a query requires code execution, direct conversational answers, or clarification on its own.
3. **No Self-Healing / Reflection Retry Loop:**
   - If `execute_code_node` crashes with an exception (`KeyError`, `ValueError`, `SyntaxError`), `workflow.py` simply passes the error text directly to `generate_insights_node`.
   - There is no feedback loop for the Code Generation Agent to read the traceback, fix the code, and re-execute.
4. **Duplicate Dataset Reading & Memory Bottlenecks:**
   - In [workflow.py:71-99](file:///c:/Users/bhanu/Downloads/Data-Analyst-Agent-20260411T135148Z-3-001/Data-Analyst-Agent/ai-data-analyst-agent/backend/agents/workflow.py#L71-L99) and [workflow.py:327-333](file:///c:/Users/bhanu/Downloads/Data-Analyst-Agent-20260411T135148Z-3-001/Data-Analyst-Agent/ai-data-analyst-agent/backend/agents/workflow.py#L327-L333), the full dataset is read from disk on every chat message **twice** per user interaction.
   - Running `.describe(include='all')` on every query introduces severe latency on large datasets.

---

### Problem Category B: Code Execution Engine Vulnerabilities & Bugs
1. **Critical Security Vulnerability (Arbitrary Code Execution):**
   - In [execution_engine.py:66](file:///c:/Users/bhanu/Downloads/Data-Analyst-Agent-20260411T135148Z-3-001/Data-Analyst-Agent/ai-data-analyst-agent/backend/execution_engine.py#L66):
     ```python
     self.allowed_globals = {
         ...
         '__builtins__': builtins,
     }
     ```
   - The engine passes the raw Python `builtins` module into `exec()`.
   - Line 89 only strips lines that begin with `import ` or `from `.
   - Anyone or any hallucinating LLM can execute arbitrary malicious code via:
     `__builtins__.__import__('os').system(...)`, `open('...', 'w').write(...)`, `eval(...)`, or `__subclasses__()`.
2. **Fragile String-Based Import Stripping:**
   - Lines starting with `import ` are stripped, which breaks multi-line imports like:
     ```python
     from sklearn.metrics import (
         accuracy_score
     )
     ```
     The closing parenthesis and inner lines remain, causing an immediate `SyntaxError`.
   - If a string literal happens to start with `import `, it is incorrectly stripped.
3. **Broken Regex Replacement for `.corr()`:**
   - In [execution_engine.py:25-46](file:///c:/Users/bhanu/Downloads/Data-Analyst-Agent-20260411T135148Z-3-001/Data-Analyst-Agent/ai-data-analyst-agent/backend/execution_engine.py#L25-L46):
     ```python
     _CORR_GUARD_RE = re.compile(r"(?P<obj>[A-Za-z_]\w*)\.corr\(")
     ```
   - If the code contains `df[['a', 'b']].corr()`, the regex matches `b.corr(` and creates invalid syntax:
     `df[['a', b.select_dtypes(include=[np.number]).corr()]`.
   - If code has `df.dropna().corr()`, it replaces `dropna.corr(` -> `df.dropna.select_dtypes(...)`, crashing with an `AttributeError`.
4. **Plotly Figure Variable Misses:**
   - In [execution_engine.py:144](file:///c:/Users/bhanu/Downloads/Data-Analyst-Agent-20260411T135148Z-3-001/Data-Analyst-Agent/ai-data-analyst-agent/backend/execution_engine.py#L144):
     It only looks for `local_vars['plotly_fig']`.
     If the LLM names the figure `fig` (the standard in Plotly documentation), the chart is completely dropped and never displayed.
5. **Thread Safety Issues in Matplotlib:**
   - `plt.clf()`, `plt.gcf()`, and `plt.close('all')` modify global process state. In an asynchronous server like FastAPI/Uvicorn, concurrent user requests will overwrite each other's plots.

---

### Problem Category C: Data Cleaning & Preprocessing Pipeline Bugs
1. **Critical Bug in Enhanced Column Type Detection:**
   - In [enhanced_data_cleaning.py:86-98](file:///c:/Users/bhanu/Downloads/Data-Analyst-Agent-20260411T135148Z-3-001/Data-Analyst-Agent/ai-data-analyst-agent/backend/pipelines/core/enhanced_data_cleaning.py#L86-L98):
     ```python
     if pd.api.types.is_numeric_dtype(series):
         unique_ratio = series.nunique() / len(series.dropna())
         if unique_ratio > 0.5 and series.nunique() > 10:
             ...
             column_types["numeric"].append(col)
         else:
             column_types["categorical"].append(col)
     ```
   - If `unique_ratio <= 0.5`, **numeric columns are classified as categorical**!
   - In real-world datasets with more than 50 rows, numeric columns like Age, Ratings (1-5), Years of Experience, or discrete counts almost always have `unique_ratio <= 0.5`.
   - As a result, standard numeric columns are incorrectly treated as categoricals, skipping numeric imputation, outlier detection, and correlation analysis.
2. **Dangerous Hardcoded Boolean Conversion:**
   - In [enhanced_preprocess.py:56-63](file:///c:/Users/bhanu/Downloads/Data-Analyst-Agent-20260411T135148Z-3-001/Data-Analyst-Agent/ai-data-analyst-agent/backend/pipelines/core/enhanced_preprocess.py#L56-L63):
     `_BOOLEAN_PATTERNS` maps `"male": True, "female": False`.
     Converting gender categories to boolean is factually inaccurate, destroys category values if non-binary entries exist, and causes data corruption.
3. **Dead Loop Bug in Legacy Preprocessing:**
   - In [preprocess.py:423-428](file:///c:/Users/bhanu/Downloads/Data-Analyst-Agent-20260411T135148Z-3-001/Data-Analyst-Agent/ai-data-analyst-agent/backend/pipelines/legacy/preprocess.py#L423-L428):
     ```python
     bool_cols = [c for c in X.columns if X[c].dtype in ["bool", "boolean"] or pd.api.types.is_bool_dtype(X[c])]
     for col in bool_cols:
         bool_cols = [c for c in X.columns if str(X[c].dtype) in ["bool", "boolean"]]
         for col in bool_cols:
             X[col] = X[col].astype(int)
     ```
     The outer loop variable is shadowed and the inner loop runs redundantly $N \times N$ times.
4. **Architectural Duplication:**
   - The repository maintains two parallel preprocessing pipelines: `pipelines/legacy` and `pipelines/core`.
   - Endpoints in `backend/main.py` mix and match them inconsistently (`/upload-enhanced` uses core, but `/model-ready` uses legacy).
   - [modeling.py](file:///c:/Users/bhanu/Downloads/Data-Analyst-Agent-20260411T135148Z-3-001/Data-Analyst-Agent/ai-data-analyst-agent/backend/modeling.py) is orphaned dead code that is never invoked.

---

### Problem Category D: API & Backend Issues
1. **Slow Synchronous Health Checks:**
   - `/ask` triggers `get_llm()` multiple times per query, causing recurring 5-second connection check penalties to Ollama.
2. **Missing Input Validation & Error Propagation:**
   - `/ask` wraps the workflow invocation in a generic `try...except` and returns `HTTP 500` with the stringified exception, stripping diagnostic context.
3. **Dataset Storage Growth:**
   - Every file uploaded to `/upload` is written to `backend/../datasets/` with a UUID. There is no session expiration, cleanup task, or temporary file management.

---

### Problem Category E: Frontend (Streamlit) Issues
1. **Unconditional Figure Display:**
   - If `plotly_figure` or `figure` is present in the response, Streamlit always draws it, even if the user asked a conversational or meta question.
2. **Loss of Chat History Context:**
   - When building `chat_history_payload` in [streamlit_app.py:636-649](file:///c:/Users/bhanu/Downloads/Data-Analyst-Agent-20260411T135148Z-3-001/Data-Analyst-Agent/ai-data-analyst-agent/frontend/streamlit_app.py#L636-L649), if an assistant message was produced by conversational mode (no `insights` or `code`), the content is omitted or malformed.
3. **Inflexible Suggested Questions:**
   - Suggested questions are only generated on initial dataset load. If a user asks "give me more questions", there is no direct hook to refresh suggestions.

---

## 3. Proposed Solutions & Technical Blueprint

### Solution 1: Fix Intent Classification & The Guardrail (Resolves the Screenshot Issue)
1. **Expand Intent Categories:**
   Update the intent prompt in `understand_query_node` to recognize five distinct intents:
   * `VISUALIZATION`: The user explicitly requests a chart, plot, graph, heatmap, or visual distribution for specific data/columns.
   * `DATA_ONLY`: The user asks a specific quantitative/statistical calculation question (mean, count, filter, top 5, aggregation) without visual output.
   * `META_QUESTIONS`: The user is asking what questions they can ask, what plots can be made, or what can be explored on the dataset (e.g., *"what type of questions that i can ask on this dataset?"*).
   * `EXPLANATION_OR_CHAT`: The user is asking a conversational question, follow-up, or general clarification about the dataset/schema without requiring code execution.
   * `OFF_TOPIC`: The user asks something completely unrelated to the dataset (weather, poems, generic coding).

2. **Route `META_QUESTIONS` Directly to a Question Generator / Conversational Node:**
   * When `META_QUESTIONS` is detected:
     Route to `generate_suggestions_node` or `conversational_response_node`.
     Provide specific suggestions categorized by:
     - Numerical distributions & correlations
     - Categorical groupings & comparisons
     - Recommended plots tailored to the columns
   * **Bypass `generate_code_node` completely!** No code is executed, and no figure is created.

3. **Generalize the Underspecified Guardrail:**
   * Instead of only checking `wants_facet`, check if any visualization marker is present without any valid column names or clear intent (e.g., `"draw a plot"`, `"visualize something"`).
   * In such cases, trigger `request_clarification` and list the available columns and possible plot options.

4. **Suppress Unrequested Figures in Frontend and Backend:**
   * If the classified intent is `META_QUESTIONS` or `DATA_ONLY` or `EXPLANATION_OR_CHAT`, explicitly set `needs_visualization = False` and ensure `execution_result['figure']` and `execution_result['plotly_figure']` are set to `None`.

---

### Solution 2: Fix Latency (Eliminate the 21-Second Delay)
1. **Cache LLM Connectivity / Base URL Validation:**
   * Check Ollama connectivity **once** on application startup or when the user changes settings, not inside every node call.
   * Remove the blocking 5-second `requests.get` call from `get_llm()` on every agent step.
2. **Cache Dataset Summary Metadata:**
   * Cache `dataset_info`, `columns`, `numeric_columns`, and `categorical_columns` in memory or in the state so `inspect_data_node` does not reload and recalculate summary statistics on every single chat message.

---

### Solution 3: Harden Code Execution Engine (`execution_engine.py`)
1. **Sandbox Security Hardening:**
   * Remove `builtins` from `allowed_globals`.
   * Provide a sanitized builtins dictionary containing only safe primitives (`range`, `len`, `int`, `float`, `str`, `list`, `dict`, `set`, `tuple`, `bool`, `enumerate`, `zip`, `print`, `min`, `max`, `sum`, `abs`, `round`).
   * Explicitly disallow `__import__`, `open`, `eval`, `exec`, `compile`, and access to `__subclasses__`.
2. **Robust Plot Detection (Support both `fig` and `plotly_fig`):**
   * Check for any `plotly.graph_objects.Figure` instance in `local_vars` (e.g., `local_vars.get('plotly_fig') or local_vars.get('fig') or local_vars.get('figure')`).
3. **Safe AST-Based Code Transformations:**
   * Replace fragile regexes with AST transformations or provide safe wrapper methods on the pre-loaded DataFrame so that `.corr()` automatically handles numeric columns without broken string substitutions.
4. **Thread-Safe Plotting:**
   * Use isolated Matplotlib figures (`fig, ax = plt.subplots()`) instead of relying on global `plt.clf()` / `plt.gcf()`.

---

### Solution 4: Fix Data Cleaning & Preprocessing Pipelines
1. **Fix Numeric Classification in `EnhancedDataCleaner`:**
   * Replace the flawed `unique_ratio > 0.5` check. A numeric column is numeric if its pandas dtype is integer/float and it has more than a minimum number of distinct values (or is continuous).
   * High-cardinality ID checks should only look for identifier keywords or 100% unique sequential integers.
2. **Remove Inappropriate Boolean Conversions:**
   * Remove `"male": True, "female": False` from `_BOOLEAN_PATTERNS`. Categorical variables like gender should remain categoricals.
3. **Clean Up Legacy Dead Code & Unify Endpoints:**
   * Fix the nested loop in `legacy/preprocess.py`.
   * Standardize backend endpoints to use the core enhanced pipeline.
   * Remove or properly integrate orphaned modules like `modeling.py`.

---

### Solution 5: Enhance Frontend User Experience
1. **Automatic Mode Switching / Unified Chat:**
   * Remove the rigid segregation between "Analysis" and "Chat" modes. Let the backend intent classifier automatically determine whether to execute code or respond conversationally.
2. **Clean Message History & Frictionless Mode:**
   * Unify chat mode and analysis mode so users do not experience mode mismatch failures.
   * Maintain consistent message serialization in session state so conversation history accurately reflects both code executions and conversational explanations.
   * Provide a modern, clean, premium Streamlit interface with quick query suggestions, dataset overview metrics, interactive data table previews, and cleanly styled cards.

---

## 4. Universal Dataset Handling Strategy (Handling Any Structure & Datatypes Optimally)

To make this agent capable of handling virtually **any dataset** with disparate structures, encodings, and datatypes efficiently:

1. **Universal File Loader (`universal_loader.py`):**
   * **Format Support:** CSV, TSV, semicolon/pipe-delimited, Excel (`.xlsx`, `.xls`), JSON/JSONL, and Parquet.
   * **Encoding & Delimiter Auto-Detection:** Automatically detect character encoding (`utf-8`, `utf-16`, `windows-1252`, `iso-8859-1`, `latin1`) and delimiters (`,`, `;`, `\t`, `|`) with fallback engines (`c` -> `python`).
   * **Malformed Header & Row Recovery:** Graceful handling of comment rows, leading blank lines, ragged rows (`on_bad_lines='skip'`), and un-named/duplicate columns.
   * **Large Dataset Memory Optimization:** Automatic downcasting of numeric columns (`float64` -> `float32`, `int64` -> `int32`/`int16`), category typecasting for repeated low-cardinality strings, and intelligent sampling for datasets exceeding memory thresholds.

2. **Resilient Data Type Identification & Schema Profiler:**
   * True numeric detection that does not misclassify columns with repeated values (e.g. ratings, ages, discrete counts).
   * Robust datetime parsing supporting ISO formats, timestamps, epoch milliseconds, and disparate date representations without breaking on invalid rows (`errors='coerce'`).
   * Unstructured text detection vs categorical variables.
   * Geo-coordinate identification (latitude/longitude) and currency/percentage string normalization.

3. **High-Performance Dataset Cache:**
   * Cache dataset metadata and sample snapshots in memory (LRU cache / process memory) so `inspect_data` and query agents never incur redundant disk read and computation costs.

---

## 5. Implementation Roadmap (Step-by-Step Execution)

| Phase | Focus | Key Deliverables |
|---|---|---|
| **Phase 1: Intent & Guardrail Fix (Priority)** | Intent & Router | • Add `META_QUESTIONS` and `EXPLANATION` intents to prompt.<br>• Route meta questions to suggestion/conversational nodes without generating code.<br>• Enhance underspecified guardrail to catch general vague visual requests.<br>• Suppress unwanted figures when visualization is not requested. |
| **Phase 2: Performance & Latency Optimization** | Latency & Caching | • Remove redundant synchronous Ollama health check from `get_llm()`.<br>• Cache dataset inspection metadata in memory/state to avoid repeated re-reads. |
| **Phase 3: Execution Engine Security & Robustness** | Sandbox & Plotting | • Sanitize `allowed_globals` (remove raw `builtins`).<br>• Capture `fig`, `plotly_fig`, and any Plotly Figure instance.<br>• Replace broken regex `.corr()` replacement with safe DataFrame wrapper.<br>• Ensure Matplotlib thread safety. |
| **Phase 4: Universal Dataset Ingestion & Pipeline Robustness** | Universal Datasets | • Build universal multi-format loader (CSV, Excel, JSON, Parquet, TSV).<br>• Delimiter & encoding auto-detection with fallback recovery.<br>• Fix numeric column detection heuristic in `enhanced_data_cleaning.py`.<br>• Remove gender-to-boolean conversion and fix legacy nested loops.<br>• Downcast numeric types for optimal memory performance. |
| **Phase 5: Clean, Modern, Premium Streamlit UI** | Frontend & UX | • Unified, seamless chat interface (no clunky radio toggle friction).<br>• Clean typography, glassmorphic dark theme, polished message bubbles.<br>• Interactive dataset schema & preview tabs.<br>• Instant suggestion chips and export options. |
| **Phase 6: Automated Testing & Verification** | Quality Assurance | • Comprehensive pytest suite verifying: intent classification, guardrail rejection, code execution, multi-format dataset loading, and API endpoints. |

---

## 6. Strict Guardrail for Data Analysis Mode (Rejecting General Questions)

### 6.1 The Issue
* **User Queries in Data Analysis Mode:**
  1. `"what is python?"`
  2. `"generate a code on consective ones"`
* **Observed Defect:**
  - For `"what is python?"`, the small local model (`qwen2.5-coder:3b`) classified it as `EXPLANATION`, causing `conversational_response_node` to explain the Python language inside Data Analysis mode.
  - For `"generate a code on consective ones"`, the model saw the word "code" and classified it as `DATA_ONLY`/`VISUALIZATION`, causing `generate_code_node` to hallucinate Titanic dataset analysis code (survival rates and fare statistics) for an algorithm question.

### 6.2 The Fix
1. **Deterministic Explicit Off-Topic Detection (`is_explicit_off_topic_query`):**
   - Detects DSA and competitive programming patterns (`consecutive ones`, `binary search`, `fibonacci`, `reverse string`, `linked list`, `two sum`, `leetcode`, `bubble sort`, `dynamic programming`, etc.).
   - Catches generic code generation requests not mentioning dataset columns or data analysis operations (`generate a code on ...`, `write a function to ...`).
   - Detects general knowledge and encyclopedic questions (`what is python`, `what is machine learning`, `who is ...`, `what is the capital of ...`, `recipe for ...`).
   - Filters out conversational small talk and chitchat (`hi`, `hello`, `who are you`, `tell me a joke`, `write a poem`).

2. **Dataset Groundedness Verification (`is_query_dataset_grounded`):**
   - In **Data Analysis Mode** (`chat_mode == "analysis"`), any query with zero grounding to dataset columns, data structures, summary statistics, or visual operations is immediately flagged as `OFF_TOPIC`.
   - Prevents ungrounded general queries from entering code generation or conversational execution.

3. **User-Friendly Redirection in Data Analysis Mode:**
   - Routes directly to `reject_off_topic_node`.
   - Returns the exact requested redirection message:
     > *"This question doesn't look like a data analysis related question for the loaded dataset. Please switch to **💬 Chat (Discuss Insights)** mode for general questions or conversation, or ask a data-specific question about the dataset (e.g., 'What is the average of <column>?', 'Show correlations', 'Plot distribution of <column>')."*
   - Suppresses code execution, figure generation, and metadata expanders in the UI.

---

## 7. Chart Container Sizing & Executive Stakeholder Insights

### 7.1 Chart Container Sizing (Plotly vs Matplotlib/Seaborn)
* **Problem:** Plotly heatmaps and charts appeared squished into a tiny square box in the center, whereas Matplotlib/Seaborn static images were stretched to 100% viewport width and appeared excessively large.
* **Root Causes:**
  - `px.imshow` in Plotly enforces a square aspect ratio (`scaleanchor='y', constrain='domain'`), restricting width to a tiny box when few columns exist.
  - Streamlit's `st.plotly_chart` was passed `width='stretch'` instead of `use_container_width=True`.
  - Matplotlib figures were saved without dimension caps and rendered with `width='stretch'`.
* **Fixes Applied:**
  - In `execution_engine.py` and `streamlit_app.py`: Set `autosize=True, height=520, margin=dict(l=40, r=40, t=50, b=50)` and cleared `scaleanchor` and `constrain` on heatmap axes so Plotly figures comfortably fill the chat container.
  - In `execution_engine.py`: Normalized Matplotlib figures to (9.5, 5.0) inches at 110 dpi.
  - In `streamlit_app.py`: Added CSS rules `.stPlotlyChart { width: 100% !important; min-height: 520px !important; }` and `[data-testid="stImage"] img { max-height: 520px !important; width: auto !important; margin: auto; }`, ensuring both formats render with balanced, consistent ~520px dimensions.

### 7.2 Executive Stakeholder Insights (3-5 Lines Max)
* **Problem:** LLMs generated generic textbook explanations (e.g. *"The heatmap shows correlation between features... The color scale is 1 to 1"*).
* **Fixes Applied:**
  - In `execution_engine.py`: Injected exact statistical findings (top correlations, descriptive statistics) into execution output so the LLM always has real data numbers to cite.
  - In `workflow.py` (`generate_insights_node`): Redesigned the prompt for an executive stakeholder audience. Demands 2-3 data-driven findings in 3-5 lines max with real metrics and directs the user to **💬 Chat (Discuss Insights)** mode for follow-up questions.
  - In `main.py` (`/chat-stream`) and `workflow.py` (`conversational_response_node`): Integrated the cached dataset profiling snippet and consultation guidelines so stakeholders can ask follow-ups in Chat mode and receive clear, in-depth explanations.

---

## 8. Hybrid Guardrail for Actionable vs Conceptual/Definition Queries ("what is bar chart")

### 8.1 The Problem
* **User Query:** `"what is bar chart"` typed in **Data Analysis (Code & Charts)** mode.
* **Bug Observed:** The system treated `"bar chart"` as a visualization trigger, generated unwanted code (`px.bar(df, x='survived')`), rendered a misleading bar chart on the dataset index, and generated hallucinated executive findings.
* **Root Cause:** A naive deterministic substring check in `workflow.py` matched any string containing `"bar chart"`, immediately forcing `intent = "VISUALIZATION"` without checking whether the query was an actionable command (`"plot a bar chart of class"`) or an interrogative conceptual inquiry (`"what is bar chart"`).

### 8.2 The Hybrid Guardrail Architecture
Instead of relying solely on brittle regexes or unconstrained LLM hallucinations, a multi-layer hybrid approach is implemented:

1. **Linguistic Intent Discrimination (`is_conceptual_or_definition_query`):**
   - Detects conceptual/educational queries asking for chart definitions, statistical concepts, or formulas (`what is (a )?(bar chart|histogram|scatter plot|box plot|heatmap|correlation|...)`, `define ...`, `explain ...`).
   - Ensures pure definition questions are flagged as conceptual, NOT plotting requests.

2. **Actionable Plotting Gate (`is_actionable_visualization_request`):**
   - Requires an active visualization command verb (`plot`, `draw`, `visualize`, `show me`, `create`, `render`, `display a chart`) OR a dataset-column-mapped phrase (`bar chart of <column>`).
   - Explicitly rejects queries flagged by `is_conceptual_or_definition_query`.

3. **LLM Semantic Classifier with Hybrid Post-Validation:**
   - In `understand_query_node`, the LLM prompt explicitly classifies pure definitions as `CONCEPTUAL_QUESTION`.
   - In Data Analysis mode, a post-validation safety gate enforces that if an LLM hallucinates `VISUALIZATION` on a conceptual definition question, it is overridden to `OFF_TOPIC`.

4. **Transparent Redirection to Chat Mode:**
   - The user is notified with a polite message to switch to **💬 Chat (Discuss Insights)** mode to learn about chart types and concepts.
   - Code execution and plotting are completely suppressed.


