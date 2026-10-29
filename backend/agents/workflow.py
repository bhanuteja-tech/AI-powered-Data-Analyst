import logging
import os
import re
import time
from pathlib import Path
from typing import Dict, Any

from dotenv import load_dotenv

# Load .env from project root
_ENV_PATH = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(_ENV_PATH)

logger = logging.getLogger(__name__)

import pandas as pd
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from backend.agents.state import AgentState
from backend.execution_engine import ExecutionEngine
from backend.universal_loader import load_dataset, get_dataset_profiling_snippet

execution_engine = ExecutionEngine()

# Cache for Ollama reachability check: url -> (is_reachable: bool, timestamp: float)
_OLLAMA_CACHE: Dict[str, tuple[bool, float]] = {}



def get_llm(state: AgentState):
    provider = state.get("provider", "openai")
    model = state.get("model_name", "gpt-4o-mini")
    api_key = state.get("api_key", "")

    if provider == "openrouter":
        return ChatOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key or os.environ.get("OPENROUTER_API_KEY", "dummy"),
            model=model,
            temperature=0,
        )
    elif provider == "ollama":
        ollama_base_url = (
            state.get("ollama_base_url")
            or os.environ.get("OLLAMA_BASE_URL")
            or "http://localhost:11434/v1"
        )

        # Cached reachability check: check at most once every 60 seconds
        _check_url = ollama_base_url.rstrip("/")
        if _check_url.endswith("/v1"):
            _check_url = _check_url[:-3]

        now = time.time()
        cached = _OLLAMA_CACHE.get(_check_url)
        is_reachable = False
        if cached and (now - cached[1] <= 60):
            is_reachable = cached[0]
        else:
            import requests as _requests
            try:
                _headers = {"ngrok-skip-browser-warning": "true"}
                _resp = _requests.get(f"{_check_url}/api/tags", timeout=4, headers=_headers)
                if _resp.status_code == 200:
                    is_reachable = True
                    _OLLAMA_CACHE[_check_url] = (True, now)
                else:
                    _OLLAMA_CACHE[_check_url] = (False, now)
            except Exception:
                _OLLAMA_CACHE[_check_url] = (False, now)

        if not is_reachable:
            # Automatic fallback to local Ollama if remote/ngrok URL is down
            if "localhost" not in _check_url and "127.0.0.1" not in _check_url:
                try:
                    import requests as _requests
                    if _requests.get("http://localhost:11434/api/tags", timeout=2).status_code == 200:
                        logger.info(f"Ollama at {_check_url} unreachable, falling back to local http://localhost:11434/v1")
                        return ChatOpenAI(
                            base_url="http://localhost:11434/v1",
                            api_key="ollama",
                            model=model,
                            temperature=0,
                        )
                except Exception:
                    pass
            raise RuntimeError(
                f"Cannot reach Ollama at {ollama_base_url}. "
                "Please ensure Ollama is running or switch provider to 'openrouter'."
            )

        return ChatOpenAI(
            base_url=ollama_base_url,
            api_key="ollama",
            model=model,
            temperature=0,
        )
    else:  # openai
        return ChatOpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY", "dummy"),
            model=model,
            temperature=0,
        )


def inspect_data_node(state: AgentState) -> AgentState:
    """Analyzes the dataset using universal loader and extracts schema info."""
    dataset_path = state["dataset_path"]
    df, meta = load_dataset(dataset_path, use_cache=True)

    state["dataset_info"] = get_dataset_profiling_snippet(df)
    state["columns"] = df.columns.tolist()
    state["numeric_columns"] = df.select_dtypes(include=["number"]).columns.tolist()
    state["categorical_columns"] = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    return state


def is_valid_suggested_question(text: str, columns: list = None) -> bool:
    """Validates that a string is actually a question or analysis command, not descriptive preamble."""
    if not text or len(text) < 10 or len(text) > 180:
        return False

    t_lower = text.lower()

    # Reject conversational preambles and dataset descriptions
    disallowed_phrases = [
        "dataset", "breakdown", "key features", "commonly used",
        "here are", "here's", "you can ask", "provided is", "subset of",
        "for regression", "for classification", "consists of", "dwelling",
        "unique identifier", "target variable is"
    ]
    if any(phrase in t_lower for phrase in disallowed_phrases):
        return False

    # Reject "Column: description of column" format (e.g. "Id: A unique identifier")
    if re.match(r"^[A-Za-z0-9_]+\s*:\s+[A-Z]", text) and not text.endswith("?"):
        return False

    # Valid questions must either end with '?' or start with an analytical command
    valid_verbs = (
        "what", "how", "which", "is", "are", "can", "why",
        "plot", "show", "compare", "calculate", "find", "distribution",
        "scatter", "histogram", "bar", "box", "heatmap", "correlation"
    )
    return text.endswith("?") or t_lower.startswith(valid_verbs)


def generate_grounded_questions(
    columns: list,
    numeric_columns: list = None,
    categorical_columns: list = None,
    max_questions: int = 5,
) -> list[str]:
    """Generates deterministic, high-value questions directly grounded in the dataset schema."""
    cols = [c for c in (columns or []) if isinstance(c, str) and c]
    num_cols = [c for c in (numeric_columns or []) if isinstance(c, str) and c]
    cat_cols = [c for c in (categorical_columns or []) if isinstance(c, str) and c]

    id_terms = ["id", "index", "row_id", "serial", "unnamed", "guid", "uuid"]
    meaningful_nums = [c for c in num_cols if not any(term in c.lower() for term in id_terms)]
    candidate_nums = meaningful_nums if meaningful_nums else num_cols

    meta_cat_terms = ["id", "name", "ticket", "cabin", "url", "description", "uuid", "row_id"]
    meaningful_cats = [c for c in cat_cols if not any(term in c.lower() for term in meta_cat_terms)]
    candidate_cats = meaningful_cats if meaningful_cats else cat_cols

    # Prioritize key target/metric columns if present
    priority_keywords = ["price", "sale", "cost", "target", "survived", "amount", "salary", "profit", "fare", "revenue"]
    priority_nums = [c for c in candidate_nums if any(pk in c.lower() for pk in priority_keywords)]
    if priority_nums:
        primary_num = priority_nums[0]
        other_nums = [c for c in candidate_nums if c != primary_num]
        candidate_nums = [primary_num] + other_nums
    elif candidate_nums and len(candidate_nums) > 1:
        # Often the last numerical column is the primary target (e.g. SalePrice)
        last_num = candidate_nums[-1]
        other_nums = candidate_nums[:-1]
        candidate_nums = [last_num] + other_nums

    questions: list[str] = []

    if candidate_nums:
        questions.append(f"Plot the distribution of {candidate_nums[0]}")
    if candidate_nums and candidate_cats:
        questions.append(f"What is the average {candidate_nums[0]} grouped by {candidate_cats[0]}?")
    if len(candidate_nums) >= 2:
        questions.append(f"Scatter plot of {candidate_nums[1]} vs {candidate_nums[0]}")
    if candidate_cats:
        questions.append(f"Plot a bar chart of {candidate_cats[0]}")
    if len(candidate_nums) >= 2:
        questions.append("Show a correlation heatmap across all numerical columns")
    elif candidate_nums:
        questions.append(f"What is the median, minimum, and maximum of {candidate_nums[0]}?")

    unique_q: list[str] = []
    seen = set()
    for q in questions:
        if q and q not in seen:
            seen.add(q)
            unique_q.append(q)

    return unique_q[:max_questions]


def generate_suggestions_node(state: AgentState) -> AgentState:
    """Generates suggested questions based on dataset schema when query is empty."""
    cols = state.get("columns", [])
    num_cols = state.get("numeric_columns", [])
    cat_cols = state.get("categorical_columns", [])

    default_questions = generate_grounded_questions(cols, num_cols, cat_cols, max_questions=4)

    try:
        llm = get_llm(state)
        sample_nums = [c for c in num_cols if not any(x in c.lower() for x in ["id", "unnamed"])][:8]
        sample_cats = [c for c in cat_cols if not any(x in c.lower() for x in ["id", "uuid"])][:8]

        sys_prompt = (
            "You are an AI Data Analyst Assistant.\n"
            "Your task is to suggest 3 to 4 analytical questions the user can ask to explore this dataset.\n"
            "STRICT RULES:\n"
            "1. Output ONLY 3 or 4 numbered questions (1., 2., 3.).\n"
            "2. Each question MUST be an actual analytical request or end with '?'.\n"
            "3. Use exact column names from the provided schema.\n"
            "4. NEVER describe what the dataset is or what columns mean.\n"
            "5. NEVER include introductory remarks, greetings, notes, or bullet points explaining fields.\n"
            "Examples:\n"
            "1. What is the average SalePrice grouped by Neighborhood?\n"
            "2. Plot the distribution of GrLivArea.\n"
            "3. Scatter plot of OverallQual vs SalePrice.\n"
            "4. Show a correlation heatmap across all numerical columns."
        )

        user_prompt = (
            f"DATASET SCHEMA:\n"
            f"Numerical Columns: {', '.join(sample_nums)}\n"
            f"Categorical Columns: {', '.join(sample_cats)}"
        )

        messages = [
            ("system", sys_prompt),
            ("user", user_prompt)
        ]

        result = llm.invoke(messages)

        extracted_questions = []
        for line in result.content.split("\n"):
            line = line.strip()
            cleaned = re.sub(r"^(\d+[\.\)]|\*|\-)\s*", "", line).strip()
            if is_valid_suggested_question(cleaned, cols):
                extracted_questions.append(cleaned)

        if len(extracted_questions) >= 3:
            state["suggested_questions"] = extracted_questions[:4]
            return state
        elif extracted_questions:
            combined = list(extracted_questions)
            for dq in default_questions:
                if dq not in combined:
                    combined.append(dq)
                if len(combined) >= 3:
                    break
            state["suggested_questions"] = combined[:4]
            return state
    except Exception as e:
        logger.warning(f"Error generating LLM suggestions: {e}")

    state["suggested_questions"] = default_questions[:4]
    return state


def split_identifier_words(name: str) -> list[str]:
    """Splits PascalCase, camelCase, snake_case, kebab-case, and dot-separated identifiers into individual words."""
    if not name or not isinstance(name, str):
        return []
    s = re.sub(r'([a-z0-9])([A-Z])', r'\1 \2', name)
    s = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', s)
    s = re.sub(r'[^a-zA-Z0-9]+', ' ', s)
    return [w.lower() for w in s.split() if w]


def normalize_token(token: str) -> str:
    """Normalizes simple English plurals for robust fuzzy matching."""
    t = token.lower().strip()
    if len(t) > 4 and t.endswith("ies"):
        return t[:-3] + "y"
    if len(t) > 3 and t.endswith("es") and not t.endswith("ses"):
        return t[:-2]
    if len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
        return t[:-1]
    return t


COLUMN_SYNONYM_LOOKUP: dict[str, set[str]] = {
    # Time
    "yr": {"year", "yr", "years", "time", "date"},
    "yrs": {"year", "yr", "years", "time", "date"},
    "year": {"year", "yr", "years"},
    "years": {"year", "yr", "years"},
    "mo": {"month", "mo", "months"},
    "mos": {"month", "mo", "months"},
    "month": {"month", "mo", "months"},
    "months": {"month", "mo", "months"},
    "date": {"day", "date", "time", "period"},
    # Real estate & economics
    "price": {"price", "cost", "fare", "pricing", "prices", "val", "value", "saleprice", "amount"},
    "prices": {"price", "cost", "fare", "pricing", "prices", "val", "value", "saleprice", "amount"},
    "sale": {"sale", "sales", "sold", "transaction"},
    "sales": {"sale", "sales", "sold", "transaction"},
    "sold": {"sold", "sale", "sales"},
    "saleprice": {"price", "cost", "fare", "pricing", "value", "saleprice", "sale price", "house price", "home price"},
    "house": {"home", "housing", "property", "building", "dwelling", "house", "houses"},
    "houses": {"home", "housing", "property", "building", "dwelling", "house", "houses"},
    "area": {"area", "size", "sf", "sqft", "square feet", "acreage"},
    "sf": {"area", "size", "sf", "sqft", "square feet"},
    "sqft": {"area", "size", "sf", "sqft", "square feet"},
    "blt": {"built", "build", "blt"},
    "built": {"built", "build", "blt"},
    "remod": {"remodel", "remodeled", "renovated", "renovation"},
    "bsmt": {"basement", "bsmt"},
    "basement": {"basement", "bsmt"},
    "gar": {"garage", "gar"},
    "garage": {"garage", "gar"},
    "flr": {"floor", "flr", "level"},
    "floor": {"floor", "flr", "level"},
    "liv": {"living", "liv"},
    "living": {"living", "liv"},
    "rm": {"room", "rooms", "rm"},
    "rms": {"room", "rooms", "rms"},
    "room": {"room", "rooms", "rm"},
    "rooms": {"room", "rooms", "rms"},
    "bed": {"bedroom", "bedrooms", "bed", "beds"},
    "bedroom": {"bedroom", "bedrooms", "bed", "beds"},
    "bedrooms": {"bedroom", "bedrooms", "bed", "beds"},
    "bath": {"bathroom", "bathrooms", "bath", "baths"},
    "baths": {"bathroom", "bathrooms", "bath", "baths"},
    "bathroom": {"bathroom", "bathrooms", "bath", "baths"},
    "bathrooms": {"bathroom", "bathrooms", "bath", "baths"},
    "qual": {"quality", "qual", "grade", "score", "rating"},
    "quality": {"quality", "qual", "grade", "score", "rating"},
    "cond": {"condition", "cond", "state", "status"},
    "condition": {"condition", "cond", "state", "status"},
    "nbrhd": {"neighborhood", "neighbourhood", "area", "location"},
    "neighborhood": {"neighborhood", "neighbourhood", "area", "location", "suburb"},
    "neighbourhood": {"neighborhood", "neighbourhood", "area", "location", "suburb"},
    "zone": {"zoning", "zone", "district"},
    "zoning": {"zoning", "zone", "district"},
    "subclass": {"subclass", "class", "category"},
    "val": {"value", "val", "amount", "worth"},
    "value": {"value", "val", "amount", "worth"},
    "amt": {"amount", "amt", "value", "sum"},
    "amount": {"amount", "amt", "value", "sum"},
    "qty": {"quantity", "qty", "count", "number"},
    "quantity": {"quantity", "qty", "count", "number"},
    "num": {"number", "num", "count"},
    "cnt": {"count", "cnt", "number"},
    "pct": {"percent", "percentage", "pct", "ratio"},
    "percent": {"percent", "percentage", "pct", "ratio"},
    "percentage": {"percent", "percentage", "pct", "ratio"},
    "pop": {"population", "pop", "inhabitants"},
    "temp": {"temperature", "temp"},
    "dist": {"distance", "dist", "length"},
    "cust": {"customer", "client", "cust"},
    "prod": {"product", "item", "prod"},
    "emp": {"employee", "worker", "staff", "emp"},
    "dept": {"department", "dept", "division"},
    "sal": {"salary", "wage", "compensation", "sal"},
    "salary": {"salary", "wage", "compensation", "sal"},
    "rev": {"revenue", "turnover", "sales", "rev"},
    "revenue": {"revenue", "turnover", "sales", "rev"},
    "car": {"car", "cars", "vehicle", "vehicles"},
    "cars": {"car", "cars", "vehicle", "vehicles"},
    "lot": {"lot", "land", "parcel"},
    "survived": {"survived", "survival", "survive", "survivors"},
    "fare": {"fare", "price", "ticket", "cost"},
    "age": {"age", "old", "years old"},
    "sex": {"sex", "gender", "male", "female"},
}

VIZ_AND_CALC_TERMS = {
    "plot", "plots", "plotting", "chart", "charts", "graph", "graphs",
    "draw", "render", "display", "visualize", "show", "calculate", "calc",
    "compute", "find", "get", "tell", "histogram", "scatter", "distribution",
    "bar", "box", "boxplot", "violin", "heatmap", "correlation", "corr",
    "pie", "line", "table", "data", "dataset", "dataframe", "df", "column",
    "columns", "feature", "features", "row", "rows"
}


def check_column_mention(query: str, columns: list) -> bool:
    """
    Checks if the query meaningfully references any dataset column.
    Supports PascalCase/camelCase/snake_case components, abbreviations,
    synonyms, and plural variations.
    """
    if not query or not columns:
        return False

    q = query.lower()
    stop_words = {"a", "an", "the", "in", "on", "at", "to", "for", "of", "by", "is", "it", "as", "or", "and", "per"}
    generic_words = {"id", "no", "type", "date", "name", "total", "value"}

    q_tokens = [re.sub(r"[^a-z0-9]", "", w) for w in q.split() if w]
    q_tokens_set = set(q_tokens)
    q_tokens_norm = {normalize_token(w) for w in q_tokens if w}

    for c in columns:
        if not c or not isinstance(c, str):
            continue
        c_clean = c.strip().lower()
        if not c_clean:
            continue

        c_pat = re.escape(c_clean).replace(r"\_", r"[\s_]")
        if c_clean in stop_words:
            if re.search(rf"\b(column\s+{c_pat}|['\"`]{c_pat}['\"`])\b", q):
                return True
            continue

        # 1. Exact full column name match
        if re.search(rf"\b{c_pat}\b", q):
            if c_clean in generic_words:
                if any(k in q for k in ["column", "feature", "variable", "distribution", "mean", "average", "unique", "count", "missing", "plot", "show", "what is the"]):
                    return True
            else:
                return True

        # 2. Identifier splitting (e.g. SalePrice -> ['sale', 'price'], MSSubClass -> ['ms', 'sub', 'class'])
        words = split_identifier_words(c)
        if len(words) > 1:
            spaced = " ".join(words)
            if spaced in q:
                return True

        # 3. Component words and synonyms
        for w in words:
            w_norm = normalize_token(w)
            if w_norm in stop_words or w_norm in generic_words or w_norm in VIZ_AND_CALC_TERMS or len(w_norm) < 2:
                continue

            # Direct token match with normalized or original query tokens
            if w in q_tokens_set or w_norm in q_tokens_norm:
                return True

            # Regex word-boundary match in query string
            if len(w_norm) >= 3 and re.search(rf"\b{re.escape(w_norm)}\b", q):
                return True

            # Abbreviation & synonym lookup
            synonyms = COLUMN_SYNONYM_LOOKUP.get(w, set()) | COLUMN_SYNONYM_LOOKUP.get(w_norm, set())
            for syn in synonyms:
                if syn in VIZ_AND_CALC_TERMS:
                    continue
                syn_norm = normalize_token(syn)
                if syn in q_tokens_set or syn_norm in q_tokens_norm:
                    return True
                if (" " in syn and syn in q) or (len(syn) >= 3 and re.search(rf"\b{re.escape(syn)}\b", q)):
                    return True

    return False


def is_truly_vague_calculation(query: str) -> bool:
    """
    Checks if a calculation request is completely empty/vague with no target metric or subject specified.
    E.g. 'what is the average?', 'calculate the mean', 'find the maximum' -> True
    'what is the average house price per year?' -> False
    """
    q = query.lower()
    calc_words = {
        "what", "is", "are", "the", "a", "an", "of", "for", "in", "on", "at", "to", "by", "from",
        "per", "with", "calculate", "calc", "find", "get", "compute", "show", "tell", "me", "give",
        "please", "can", "you", "average", "avg", "mean", "median", "mode", "sum", "total", "count",
        "max", "maximum", "min", "minimum", "highest", "lowest", "top", "bottom", "std", "variance",
        "value", "values", "metric", "variable", "variables", "number", "numbers", "stats",
        "statistics", "something", "it", "this", "all", "column", "columns", "feature", "features",
        "dataset", "data", "table", "row", "rows"
    }
    tokens = [re.sub(r"[^\w]", "", w) for w in q.split()]
    substantive_tokens = [t for t in tokens if t and t not in calc_words]
    # Truly vague if no domain/subject words exist or if remaining tokens are purely digits (e.g. "top 5")
    return len(substantive_tokens) == 0 or all(t.isdigit() for t in substantive_tokens)


def normalize_viz_terms(text: str) -> str:
    """Expands compound visualization words into separate tokens e.g. barchart -> bar chart."""
    if not text:
        return ""
    t = text.lower()
    compounds = {
        "barchart": "bar chart",
        "barcharts": "bar charts",
        "barplot": "bar plot",
        "barplots": "bar plots",
        "scatterplot": "scatter plot",
        "scatterplots": "scatter plots",
        "linechart": "line chart",
        "linecharts": "line charts",
        "lineplot": "line plot",
        "lineplots": "line plots",
        "boxplot": "box plot",
        "boxplots": "box plots",
        "boxchart": "box chart",
        "boxcharts": "box charts",
        "piechart": "pie chart",
        "piecharts": "pie charts",
        "violinplot": "violin plot",
        "violinplots": "violin plots",
        "countplot": "count plot",
        "countplots": "count plots",
        "countchart": "count chart",
        "countcharts": "count charts",
        "pairplot": "pair plot",
        "pairplots": "pair plots",
        "heatplot": "heat plot",
        "heatplots": "heat plots",
        "distchart": "distribution chart",
        "distplot": "distribution plot",
    }
    for comp, expanded in compounds.items():
        t = re.sub(rf"\b{comp}\b", expanded, t)
    return t


def is_truly_vague_visualization(query: str) -> bool:
    """
    Checks if a visualization request is completely vague with no target column or entity specified.
    E.g. 'plot a chart', 'plot a barchart', 'draw a graph', 'show distribution' -> True
    'plot house prices', 'show distribution of prices over years' -> False
    """
    q = normalize_viz_terms(query).lower()
    viz_words = {
        "what", "is", "are", "the", "a", "an", "of", "for", "in", "on", "at", "to", "by", "from",
        "per", "with", "show", "tell", "me", "give", "please", "can", "you", "plot", "plots",
        "plotting", "chart", "charts", "graph", "graphs", "visualize", "visualization",
        "visualizations", "draw", "render", "display", "make", "create", "distribution",
        "distributions", "histogram", "histograms", "hist", "scatter", "scatterplot", "scatterplots",
        "bar", "bars", "barchart", "barcharts", "barplot", "barplots", "box", "boxplot", "boxplots",
        "boxchart", "boxcharts", "violin", "violinplot", "violinplots", "pie", "piechart", "piecharts",
        "density", "kde", "line", "lineplot", "lineplots", "linechart", "linecharts", "countplot",
        "countplots", "countchart", "countcharts", "pairplot", "pairplots", "heatmap", "heatmaps",
        "something", "it", "this", "all", "column", "columns", "feature", "features", "dataset",
        "data", "table", "row", "rows", "diagram", "diagrams", "figure", "figures", "image", "images",
        "pic", "picture"
    }
    tokens = [re.sub(r"[^\w]", "", w) for w in q.split()]
    substantive_tokens = [t for t in tokens if t and t not in viz_words]
    return len(substantive_tokens) == 0 or all(t.isdigit() for t in substantive_tokens)



def is_whole_dataset_visualization(query: str) -> bool:
    """Checks if query is a visualization that applies across the entire dataset without needing a column."""
    q = query.strip().lower()
    return (
        "heatmap" in q
        or ("correlation" in q and any(w in q for w in ["matrix", "plot", "chart", "map", "table"]))
        or "pairplot" in q
        or "pair plot" in q
        or "scatter matrix" in q
    )


def generate_clarification_for_ambiguity(
    query: str,
    columns: list,
    numeric_columns: list = None,
    categorical_columns: list = None,
) -> tuple[str, list[str]]:
    """
    Builds an articulate, high-value clarification response when a query is ambiguous.
    Explains the ambiguity clearly and provides a prominent 'These are some questions you may ask'
    section with grounded, insightful questions tailored to the dataset.
    """
    q = (query or "").strip().lower()
    cols = [c for c in (columns or []) if isinstance(c, str) and c]
    num_cols = [c for c in (numeric_columns or []) if isinstance(c, str) and c]
    cat_cols = [c for c in (categorical_columns or []) if isinstance(c, str) and c]

    if not num_cols and not cat_cols and cols:
        num_cols = cols[:4]
        cat_cols = cols[4:8]

    # Specific ambiguity rationale
    if "facet" in q or "facetgrid" in q:
        clarification_intro = (
            "Your request asks for a **facet grid**, but does not specify which variables to plot on the axes and which column to facet (group) by."
        )
    elif "distribution" in q or "histogram" in q or "density" in q or "kde" in q:
        clarification_intro = (
            "Your request asks to **plot a distribution**, but does not specify which variable or column you would like to visualize."
        )
    elif "bar" in q:
        clarification_intro = (
            "Your request asks to **plot a bar chart**, but does not specify which categorical column or feature you would like to compare."
        )
    elif "box" in q:
        clarification_intro = (
            "Your request asks to **plot a box plot**, but does not specify which numerical feature or grouping column to analyze."
        )
    elif "scatter" in q:
        clarification_intro = (
            "A **scatter plot** requires two numerical variables (X and Y), but your request does not specify which variables to compare."
        )
    elif any(ck in q for ck in ["average", "mean", "median", "sum", "max", "min", "highest", "lowest"]):
        clarification_intro = (
            f"Your request asks for a calculation ('{query.strip()}'), but does not specify which column or metric to calculate."
        )
    else:
        clarification_intro = (
            f"Your request ('{query.strip()}') is ambiguous because it does not specify which column or variable from the dataset you would like to analyze."
        )

    num_display = ", ".join(f"`{c}`" for c in num_cols[:8]) if num_cols else "None"
    cat_display = ", ".join(f"`{c}`" for c in cat_cols[:8]) if cat_cols else "None"

    # Prioritize meaningful analytical columns over generic ID/identifier columns
    id_terms = ["id", "index", "row_id", "serial", "unnamed"]
    meaningful_num_cols = [c for c in num_cols if not any(term in c.lower() for term in id_terms)]
    candidate_nums = meaningful_num_cols if meaningful_num_cols else num_cols

    meta_cat_terms = ["id", "name", "ticket", "cabin", "url", "description", "uuid"]
    meaningful_cat_cols = [c for c in cat_cols if not any(term in c.lower() for term in meta_cat_terms)]
    candidate_cats = meaningful_cat_cols if meaningful_cat_cols else cat_cols

    # Curate high-quality suggested questions tailored to the dataset
    suggested: list[str] = []

    if "distribution" in q or "histogram" in q:
        for c in candidate_nums[:2]:
            suggested.append(f"Plot the distribution of {c}")
        if candidate_cats:
            suggested.append(f"Plot a bar chart of {candidate_cats[0]}")
        if candidate_nums and candidate_cats:
            suggested.append(f"Compare {candidate_nums[0]} distribution across {candidate_cats[0]}")
        suggested.append("Show a correlation heatmap across all numerical columns")
    elif "bar" in q:
        for c in candidate_cats[:2]:
            suggested.append(f"Plot a bar chart of {c}")
        if candidate_nums and candidate_cats:
            suggested.append(f"What is the average {candidate_nums[0]} grouped by {candidate_cats[0]}?")
        if candidate_nums:
            suggested.append(f"Plot the distribution of {candidate_nums[0]}")
    elif "scatter" in q:
        if len(candidate_nums) >= 2:
            suggested.append(f"Scatter plot of {candidate_nums[0]} vs {candidate_nums[1]}")
        if len(candidate_nums) >= 3:
            suggested.append(f"Scatter plot of {candidate_nums[0]} vs {candidate_nums[2]}")
        if candidate_nums and candidate_cats:
            suggested.append(f"Scatter plot of {candidate_nums[0]} vs {candidate_nums[1] if len(candidate_nums) > 1 else candidate_nums[0]} colored by {candidate_cats[0]}")
        suggested.append("Show a correlation heatmap across all numerical columns")
    elif "facet" in q:
        if len(candidate_nums) >= 2 and candidate_cats:
            suggested.append(f"Facet grid of {candidate_nums[0]} vs {candidate_nums[1]} faceted by {candidate_cats[0]}")
        if candidate_nums and candidate_cats:
            suggested.append(f"Histogram of {candidate_nums[0]} faceted by {candidate_cats[0]}")
        suggested.append("Show a correlation heatmap across all numerical columns")
    elif any(ck in q for ck in ["average", "mean", "median", "sum", "max", "min"]):
        for c in candidate_nums[:2]:
            suggested.append(f"What is the average and median of {c}?")
        if candidate_nums and candidate_cats:
            suggested.append(f"What is the average {candidate_nums[0]} by {candidate_cats[0]}?")
        if candidate_nums:
            suggested.append(f"Plot the distribution of {candidate_nums[0]}")
    else:
        if candidate_nums:
            suggested.append(f"Plot the distribution of {candidate_nums[0]}")
        if len(candidate_nums) >= 2:
            suggested.append(f"Scatter plot of {candidate_nums[0]} vs {candidate_nums[1]}")
        if candidate_cats:
            suggested.append(f"Plot a bar chart of {candidate_cats[0]}")
        if candidate_nums and candidate_cats:
            suggested.append(f"Compare {candidate_nums[0]} across {candidate_cats[0]}")
        suggested.append("Show a correlation heatmap across all numerical columns")

    unique_suggested = []
    seen = set()
    for sq in suggested:
        if sq and sq not in seen:
            seen.add(sq)
            unique_suggested.append(sq)
    final_suggested = unique_suggested[:5]

    questions_markdown = "\n".join(f"- 🔹 `{sq}`" for sq in final_suggested)

    message = (
        f"### 🤔 Clarification Needed\n\n"
        f"{clarification_intro}\n\n"
        f"**Available Columns:**\n"
        f"- 🔢 **Numerical variables:** {num_display}\n"
        f"- 🏷️ **Categorical variables:** {cat_display}\n\n"
        f"---\n\n"
        f"### 💡 These are some questions you may ask:\n"
        f"{questions_markdown}\n\n"
        f"*(You can copy or ask any of the suggested questions above, or specify which column to analyze!)*"
    )

    return message, final_suggested


def is_conceptual_or_definition_query(query: str, columns: list) -> bool:
    """
    Detects whether a query is asking for an educational definition, concept explanation,
    or theoretical question (e.g. 'what is bar chart', 'what is a histogram', 'what is standard deviation')
    rather than commanding an analysis or visualization on the loaded dataset.
    """
    q = query.strip().lower()

    # Chart types and statistical metrics
    chart_and_stats_terms = (
        r"(bar\s*chart|histogram|scatter\s*plot|box\s*plot|boxplot|pie\s*chart|line\s*chart|"
        r"heatmap|violin\s*plot|pair\s*plot|distribution\s*plot|correlation|covariance|"
        r"standard\s*deviation|variance|median|mode|mean|iqr|outliers?|p-value|r-squared|"
        r"regression|classification|machine\s*learning|data\s*science|data\s*analysis|"
        r"chart|graph|plot|diagram)"
    )

    concept_patterns = [
        rf"^what\s+(is|are|does)\s+(a\s+|an\s+|the\s+)?{chart_and_stats_terms}\b",
        rf"^what\s+(is|are)\s+{chart_and_stats_terms}\b",
        rf"^(define|definition\s+of|meaning\s+of)\s+(a\s+|an\s+|the\s+)?{chart_and_stats_terms}\b",
        rf"^explain\s+(what\s+is\s+a\s+|what\s+is\s+|the\s+concept\s+of\s+|how\s+a\s+)?{chart_and_stats_terms}\b",
        rf"^how\s+(does|do)\s+(a\s+|an\s+)?{chart_and_stats_terms}\s+work\b",
        rf"^why\s+(do\s+we\s+use|use)\s+(a\s+|an\s+)?{chart_and_stats_terms}\b",
        rf"^when\s+(to\s+use|should\s+i\s+use)\s+(a\s+|an\s+)?{chart_and_stats_terms}\b",
        rf"^difference\s+between\s+{chart_and_stats_terms}\b",
    ]

    for pat in concept_patterns:
        if re.search(pat, q):
            # If the query mentions any dataset column, it is an analysis request, not a concept definition!
            if check_column_mention(q, columns):
                return False
            cols_clean = [str(c).lower().replace("_", " ") for c in (columns or []) if c]
            has_col = any(c in q for c in cols_clean if len(c) > 2 and c not in ["id", "no", "type"])
            if has_col and any(action_phrase in q for action_phrase in [" of ", " for ", " between "]):
                return False
            return True

    return False


def is_actionable_visualization_request(query: str, columns: list) -> bool:
    """
    Checks if query contains an active intent to render/create an unambiguous visualization.
    Must either mention a specific dataset column or be a whole-dataset chart (e.g. correlation heatmap).
    """
    q = normalize_viz_terms(query).strip().lower()

    # If it's a conceptual definition query, it is NOT an action to visualize!
    if is_conceptual_or_definition_query(q, columns):
        return False

    # Whole-dataset charts do not require specifying a column
    if is_whole_dataset_visualization(q):
        return True

    has_col = check_column_mention(q, columns)
    if not has_col:
        # If no column is mentioned, it is ambiguous, NOT actionable!
        return False

    # Check for plotting action verbs
    action_verbs = [
        r"\b(plot|draw|create|generate|visualize|render|display|make)\b",
        r"\b(show\s+me|show)\b",
        r"\b(give\s+me\s+a\s+plot|give\s+me\s+a\s+chart|build\s+a\s+chart)\b",
    ]
    has_action = any(re.search(av, q) for av in action_verbs)

    # Chart types
    chart_types = [
        "histogram", "scatter", "bar chart", "bar",
        "boxplot", "box plot", "pie chart", "line plot", "line chart",
        "violin plot", "distribution plot", "distribution", "kde", "density"
    ]
    has_chart_type = any(ct in q for ct in chart_types)

    # Action verb or chart type with the mentioned column
    if has_action or has_chart_type:
        return True

    return False


def is_explicit_off_topic_query(query: str, columns: list) -> tuple[bool, str]:
    """Identifies generic programming, DSA, general knowledge, or conversational prompts."""
    q = query.strip().lower()

    # 1. Conceptual definition queries (e.g. "what is bar chart", "what is a histogram")
    if is_conceptual_or_definition_query(q, columns):
        return True, "conceptual_definition"

    # 2. DSA / Generic algorithms / LeetCode / competitive programming
    dsa_patterns = [
        r"\b(consec[a-z]*\s+ones?)\b",
        r"\b(binary\s+search|breadth\s+first|depth\s+first|dfs|bfs)\b",
        r"\b(fibonacci|two\s+sum|three\s+sum|palindrome|anagram)\b",
        r"\b(linked\s+list|reverse\s+string|reverse\s+linked\s+list)\b",
        r"\b(bubble\s+sort|merge\s+sort|quick\s+sort|insertion\s+sort|heap\s+sort)\b",
        r"\b(dynamic\s+programming|knapsack|dijkstra|longest\s+common\s+subsequence)\b",
        r"\b(binary\s+tree|binary\s+search\s+tree|trie|avl\s+tree|b\+?\s+tree)\b",
        r"\b(stack\s+overflow|leetcode|hackerrank|codeforces)\b",
        r"\b(matrix\s+multiplication|prime\s+numbers?|factorial|armstrong)\b",
        r"\b(sliding\s+window|two\s+pointers?|kadane'?s?)\b",
    ]
    for pat in dsa_patterns:
        if re.search(pat, q):
            return True, "generic_algorithm"

    # 3. Generic code synthesis requests not mentioning dataset columns
    generic_code_gen = [
        r"^(can\s+you\s+)?(generate|write|give\s+me|create)\s+(a\s+)?(code|script|program|function)\s+(to|for|on)\s+",
        r"^(can\s+you\s+)?(solve|implement)\s+(the\s+)?(problem|algorithm|logic|code)\s+",
    ]
    for pat in generic_code_gen:
        if re.search(pat, q):
            cols_lower = [c.lower() for c in (columns or []) if isinstance(c, str)]
            data_terms = ["df", "dataset", "column", "row", "plot", "chart", "graph", "histogram", "scatter", "heatmap", "correlation", "average", "mean", "median", "missing", "null"]
            has_col = any(c in q for c in cols_lower if len(c) > 1)
            has_data_term = any(dt in q for dt in data_terms)
            if not has_col and not has_data_term:
                return True, "generic_code_generation"

    # 4. General knowledge / definitions / trivia questions unrelated to dataset
    general_knowledge_patterns = [
        r"^\s*what\s+is\s+(python|java|c\+\+|c#|javascript|rust|golang|html|css|react|angular|vue|docker|kubernetes|linux|git|github)\b",
        r"^\s*what\s+is\s+(machine\s+learning|deep\s+learning|artificial\s+intelligence|ai|nlp|computer\s+vision|blockchain|bitcoin|crypto|llm|chatgpt|openai)\b",
        r"^\s*(who\s+is|who\s+was|who\s+founded|who\s+created)\s+",
        r"^\s*what\s+is\s+the\s+capital\s+of\s+",
        r"^\s*what\s+is\s+the\s+weather\b",
        r"^\s*(tell\s+me\s+a\s+joke|write\s+a\s+poem|write\s+a\s+story|sing\s+a\s+song)\b",
        r"^\s*(recipe\s+for|how\s+to\s+cook|how\s+to\s+bake)\b",
        r"^\s*how\s+to\s+install\s+",
    ]
    for pat in general_knowledge_patterns:
        if re.search(pat, q):
            return True, "general_knowledge"

    # 5. Casual greetings & conversational small talk
    greetings = [
        r"^\s*(hi|hello|hey|howdy|greetings|good\s+morning|good\s+afternoon|good\s+evening)\s*[\!\?\.]*$",
        r"^\s*(who\s+are\s+you|what\s+are\s+you|what\s+is\s+your\s+name|tell\s+me\s+about\s+yourself)\s*[\!\?\.]*$",
        r"^\s*(how\s+are\s+you|how\s+do\s+you\s+do|what'?s\s+up)\s*[\!\?\.]*$",
    ]
    for pat in greetings:
        if re.search(pat, q):
            return True, "chitchat"

    return False, ""


def is_query_dataset_grounded(query: str, columns: list) -> bool:
    """Checks if a query has any grounding or relevance to the loaded dataset or data analysis."""
    q = query.strip().lower()
    if not q:
        return False

    # Check dataset columns
    if check_column_mention(q, columns):
        return True

    # Analytical, statistical, visualization & data structure keywords
    data_analysis_keywords = [
        "mean", "average", "avg", "median", "mode", "std", "standard deviation", "variance",
        "sum", "count", "min", "minimum", "max", "maximum", "highest", "lowest", "top", "bottom",
        "quantile", "percentile", "quartile", "iqr", "skew", "kurtosis", "describe", "summary",
        "ratio", "percentage", "proportion", "frequency",
        "correlation", "corr", "covariance", "relationship", "association", "trend", "pattern",
        "missing", "null", "nulls", "na", "nan", "duplicate", "duplicates", "outlier", "outliers",
        "unique", "value count", "cardinality", "shape", "dimension", "rows", "columns", "features",
        "dtypes", "datatypes", "data types", "head", "tail", "sample", "dataset", "dataframe", "df",
        "table", "records", "data", "schema",
        "plot", "chart", "graph", "histogram", "scatter", "heatmap", "bar", "box plot", "boxplot",
        "pie", "line", "violin", "pairplot", "visualize", "distribution", "facet",
        "group by", "grouped by", "filter", "where", "pivot", "aggregate", "breakdown"
    ]
    for kw in data_analysis_keywords:
        if re.search(rf"\b{re.escape(kw)}\b", q):
            return True

    # Meta-questions asking what can be done on the dataset
    meta_keywords = [
        "what can i ask", "what questions", "suggest questions", "what to analyze",
        "what plots", "what charts", "ideas for analysis", "suggest some questions"
    ]
    for mk in meta_keywords:
        if mk in q:
            return True

    # Follow-ups referencing an existing chart or insight
    followup_keywords = [
        "this graph", "this chart", "this plot", "that graph", "that chart",
        "above graph", "above chart", "above plot", "the plot above",
        "previous graph", "what does this mean", "explain this"
    ]
    for fk in followup_keywords:
        if fk in q:
            return True

    return False


def understand_query_node(state: AgentState) -> AgentState:
    """Analyzes user intent with hybrid linguistic checks, deterministic guards, and LLM classification."""
    state["needs_clarification"] = False
    state["clarification_message"] = ""
    q = normalize_viz_terms(state.get("query") or "").strip().lower()
    cols = state.get("columns", []) or []
    is_analysis_mode = state.get("chat_mode", "analysis") != "chat"

    # Deterministic Guard 1: Meta-questions asking what can be asked / plotted on this dataset
    meta_markers = [
        "what type of questions", "what questions", "what kind of questions",
        "questions i can ask", "questions that i can ask", "questions can i ask",
        "what can i ask", "what should i ask", "what can you analyze",
        "suggest questions", "suggest some questions", "give me questions",
        "what can i plot", "what plots can i", "what charts can i",
        "plotting questions", "visualization ideas", "suggest plots",
        "ideas for analysis", "what can you plot", "what kind of plots",
        "what visualizations can", "what graphs can"
    ]
    if any(m in q for m in meta_markers):
        state["intent"] = "META_QUESTIONS"
        state["needs_visualization"] = False
        state["off_topic"] = False
        return state

    # Deterministic Guard 2: Conversational follow-ups referencing existing charts or insights
    followup_markers = [
        "this graph", "this chart", "this plot", "this figure",
        "that graph", "that chart", "that plot", "that figure",
        "above graph", "above chart", "above plot", "above figure",
        "the graph above", "the chart above", "the plot above",
        "previous graph", "previous chart", "previous plot",
        "what does this indicate", "what does this mean", "explain this graph",
        "interpret this graph", "interpret this chart", "interpret this plot",
    ]
    if any(m in q for m in followup_markers) or re.search(r"\b(this|that|above|previous)\b.*\b(graph|chart|plot|figure)\b", q):
        state["intent"] = "EXPLANATION"
        state["needs_visualization"] = False
        state["off_topic"] = False
        return state

    # Deterministic Guard 3: Explicit Off-Topic / Conceptual Definition Questions (e.g. "what is bar chart")
    is_off_topic, reason = is_explicit_off_topic_query(q, cols)
    if is_off_topic:
        state["intent"] = "OFF_TOPIC"
        state["needs_visualization"] = False
        state["off_topic"] = True
        if reason == "conceptual_definition":
            state["clarification_message"] = (
                f"This question appears to be a conceptual definition ('{state.get('query', '')}') rather than a data analysis request for the loaded dataset. "
                "Please switch to **💬 Chat (Discuss Insights)** mode for explanations of concepts and chart types, "
                "or specify what to analyze in this dataset (e.g., 'Plot a bar chart of <column>', 'Show correlations', 'What is the average of <column>?')."
            )
        return state

    # Deterministic Guard 4: Dataset Groundedness in Data Analysis Mode
    if is_analysis_mode and not is_query_dataset_grounded(q, cols):
        state["intent"] = "OFF_TOPIC"
        state["needs_visualization"] = False
        state["off_topic"] = True
        return state

    # Deterministic Guard 5: Vague / Underspecified / Ambiguous visualization or analysis request
    mentions_col = check_column_mention(q, cols)
    num_cols = state.get("numeric_columns", []) or []
    cat_cols = state.get("categorical_columns", []) or []
    is_whole_dataset = is_whole_dataset_visualization(q)

    # 5a. Scatter plot requires two columns
    is_scatter = "scatter" in q
    if is_scatter and not is_whole_dataset:
        mentioned_cols = [c for c in cols if check_column_mention(q, [c])]
        if len(mentioned_cols) < 2:
            state["needs_clarification"] = True
            state["intent"] = "CLARIFICATION"
            state["needs_visualization"] = False
            state["off_topic"] = False
            msg, questions = generate_clarification_for_ambiguity(
                state.get("query", ""), cols, num_cols, cat_cols
            )
            state["clarification_message"] = msg
            state["suggested_questions"] = questions
            return state

    # 5b. Facet grid requires columns
    wants_facet = ("facet" in q) or ("facetgrid" in q) or ("facet grid" in q)
    if wants_facet and not mentions_col:
        state["needs_clarification"] = True
        state["intent"] = "CLARIFICATION"
        state["needs_visualization"] = False
        state["off_topic"] = False
        msg, questions = generate_clarification_for_ambiguity(
            state.get("query", ""), cols, num_cols, cat_cols
        )
        state["clarification_message"] = msg
        state["suggested_questions"] = questions
        return state

    # 5c. Any visualization request without a column (and not whole-dataset)
    viz_terms = [
        "plot", "chart", "graph", "visualize", "draw", "render",
        "distribution", "histogram", "bar", "box", "boxplot",
        "violin", "pie", "density", "kde", "scatter", "line"
    ]
    has_viz_intent = any(re.search(rf"\b{re.escape(vt)}\b", q) for vt in viz_terms)
    if has_viz_intent and not mentions_col and not is_whole_dataset and is_truly_vague_visualization(q):
        state["needs_clarification"] = True
        state["intent"] = "CLARIFICATION"
        state["needs_visualization"] = False
        state["off_topic"] = False
        msg, questions = generate_clarification_for_ambiguity(
            state.get("query", ""), cols, num_cols, cat_cols
        )
        state["clarification_message"] = msg
        state["suggested_questions"] = questions
        return state

    # 5d. Vague calculation request without a column
    has_calc_kw = any(ck in q for ck in ["average", "mean", "median", "sum", "count", "max", "maximum", "min", "minimum", "highest", "lowest", "top", "bottom", "missing", "null", "unique"])
    calc_action = any(re.search(rf"\b{ca}\b", q) for ca in ["what is", "calculate", "find", "get", "compute", "show me", "show", "tell me"])
    if has_calc_kw and calc_action and not mentions_col and is_truly_vague_calculation(q):
        state["needs_clarification"] = True
        state["intent"] = "CLARIFICATION"
        state["needs_visualization"] = False
        state["off_topic"] = False
        msg, questions = generate_clarification_for_ambiguity(
            state.get("query", ""), cols, num_cols, cat_cols
        )
        state["clarification_message"] = msg
        state["suggested_questions"] = questions
        return state

    # Deterministic Guard 6: Actionable Visualization Command vs. Quantitative Calculation
    if is_actionable_visualization_request(q, cols):
        state["intent"] = "VISUALIZATION"
        state["needs_visualization"] = True
        state["off_topic"] = False
        return state

    if has_calc_kw and mentions_col:
        state["intent"] = "DATA_ONLY"
        state["needs_visualization"] = False
        state["off_topic"] = False
        return state

    # LLM Hybrid Intent Classification for nuanced queries
    llm = get_llm(state)
    chat_context = ""
    history = state.get("chat_history", [])
    if history:
        chat_context = "RECENT CHAT HISTORY:\n"
        for msg in history[-4:]:
            chat_context += f"{msg.get('role', 'user').upper()}: {msg.get('content', '')}\n"

    sys_prompt = f"""You are an expert AI Classifier determining user INTENT in a Data Analysis tool.
AVAILABLE DATASET COLUMNS: {', '.join(cols[:15])}

CURRENT MODE: DATA ANALYSIS MODE (Code & Charts execution).
In this mode, ONLY actionable commands to perform calculations or generate plots on THIS dataset should be executed.

Classify the user query into EXACTLY ONE category:

1. "VISUALIZATION": User explicitly commands to create, render, or plot a chart, graph, heatmap, scatter, bar, line, histogram, or box plot using dataset variables.
   (Examples: "plot a bar chart of class", "show correlation heatmap", "scatter plot age vs fare")
   CRITICAL: DO NOT choose this if the user asks what a chart is (e.g. "what is bar chart")!

2. "DATA_ONLY": User asks for a quantitative calculation (average, max, min, filter, sum, count, correlation, top 5) on dataset columns.
   (Examples: "what is the average age?", "how many passengers survived?", "show top 5 fares")

3. "META_QUESTIONS": User asks what questions they can ask, what plots can be made, or what analyses can be done on the dataset.
   (Examples: "what type of questions can I ask?", "suggest some plots for this dataset")

4. "CONCEPTUAL_QUESTION": User asks for a definition, concept explanation, or educational information (e.g. "what is bar chart?", "what is a histogram?", "explain standard deviation", "what is python?").
   In Data Analysis mode, these are NOT execution commands.

5. "OFF_TOPIC": General programming (DSA/algorithms/leetcode), chitchat, creative writing, or non-data queries.

{chat_context}

Respond with ONLY the category name."""

    messages = [
        ("system", sys_prompt),
        ("user", state["query"])
    ]

    result = llm.invoke(messages)
    intent_str = result.content.strip().upper()

    if "META" in intent_str:
        state["intent"] = "META_QUESTIONS"
        state["needs_visualization"] = False
        state["off_topic"] = False
    elif "CONCEPTUAL" in intent_str or "OFF" in intent_str:
        state["intent"] = "OFF_TOPIC"
        state["needs_visualization"] = False
        state["off_topic"] = True
        state["clarification_message"] = (
            f"This question appears to be a conceptual or general question ('{state.get('query', '')}') rather than an analysis request on the loaded dataset. "
            "Please switch to **💬 Chat (Discuss Insights)** mode for explanations of concepts and chart types, "
            "or ask a data-specific request about the dataset (e.g., 'Plot a bar chart of <column>', 'Show correlations', 'What is the average of <column>?')."
        )
    elif "VISUAL" in intent_str:
        # Hybrid validation: Ensure the query is ACTUALLY an actionable visualization command
        if is_conceptual_or_definition_query(q, cols):
            state["intent"] = "OFF_TOPIC"
            state["needs_visualization"] = False
            state["off_topic"] = True
            state["clarification_message"] = (
                f"This question appears to be a conceptual question ('{state.get('query', '')}') rather than a request to plot data. "
                "Please switch to **💬 Chat (Discuss Insights)** mode for explanations of chart types and concepts, "
                "or specify which column you want to plot (e.g., 'Plot a bar chart of <column>')."
            )
        elif not mentions_col and not is_whole_dataset_visualization(q) and is_truly_vague_visualization(q):
            state["intent"] = "CLARIFICATION"
            state["needs_clarification"] = True
            state["needs_visualization"] = False
            state["off_topic"] = False
            msg, questions = generate_clarification_for_ambiguity(
                state.get("query", ""), cols, state.get("numeric_columns", []), state.get("categorical_columns", [])
            )
            state["clarification_message"] = msg
            state["suggested_questions"] = questions
        else:
            state["intent"] = "VISUALIZATION"
            state["needs_visualization"] = True
            state["off_topic"] = False
    elif "DATA" in intent_str:
        # Hybrid validation: Ensure the query is ACTUALLY a quantitative data calculation on the dataset
        if is_conceptual_or_definition_query(q, cols) or (is_analysis_mode and not is_query_dataset_grounded(q, cols)):
            state["intent"] = "OFF_TOPIC"
            state["needs_visualization"] = False
            state["off_topic"] = True
        elif not mentions_col and has_calc_kw and is_truly_vague_calculation(q):
            state["intent"] = "CLARIFICATION"
            state["needs_clarification"] = True
            state["needs_visualization"] = False
            state["off_topic"] = False
            msg, questions = generate_clarification_for_ambiguity(
                state.get("query", ""), cols, state.get("numeric_columns", []), state.get("categorical_columns", [])
            )
            state["clarification_message"] = msg
            state["suggested_questions"] = questions
        else:
            state["intent"] = "DATA_ONLY"
            state["needs_visualization"] = False
            state["off_topic"] = False
    else:
        # If in analysis mode and query has no grounding in dataset, do not allow general explanations
        if is_analysis_mode and not is_query_dataset_grounded(q, cols):
            state["intent"] = "OFF_TOPIC"
            state["needs_visualization"] = False
            state["off_topic"] = True
        else:
            state["intent"] = "EXPLANATION"
            state["needs_visualization"] = False
            state["off_topic"] = False

    return state


def handle_meta_questions_node(state: AgentState) -> AgentState:
    """Answers inquiries about what questions or plots can be asked on the dataset without running code."""
    llm = get_llm(state)
    sys_prompt = """You are an expert Data Analyst and AI Advisor.
The user is asking what questions or visualizations they can explore on this dataset.
Based on the dataset columns, data types, and statistics, provide a structured, practical guide.

Format your response cleanly:
### 📊 Suggested Visualizations
(List 3-4 specific charts mentioning exact columns and why they are insightful)

### 🔢 Statistical & Aggregation Questions
(List 3-4 specific numeric questions, e.g. group comparisons, averages, distributions)

### 🔍 Deep-Dive Hypotheses
(List 2 interesting relationships or anomalies to check)

Be direct, engaging, and do NOT write Python code."""

    messages = [
        ("system", sys_prompt),
        ("user", f"DATASET SCHEMA & SUMMARY:\n{state.get('dataset_info', '')}\n\nUSER PROMPT: {state['query']}")
    ]

    result = llm.invoke(messages)
    state["insights"] = result.content
    state["generated_code"] = ""
    state["execution_result"] = {
        "status": "success",
        "output": "",
        "figure": None,
        "plotly_figure": None
    }
    state["needs_visualization"] = False
    return state


def reject_off_topic_node(state: AgentState) -> AgentState:
    """Rejects queries unrelated to data analysis in Data Analysis mode."""
    state["generated_code"] = ""
    state["execution_result"] = {
        "status": "success",
        "output": "",
        "figure": None,
        "plotly_figure": None
    }
    state["intent"] = "OFF_TOPIC"
    state["off_topic"] = True
    state["needs_visualization"] = False
    state["insights"] = (
        "This question doesn't look like a data analysis related question for the loaded dataset. "
        "Please switch to **💬 Chat (Discuss Insights)** mode for general questions or conversation, "
        "or ask a data-specific question about the dataset (e.g., 'What is the average of <column>?', 'Show correlations', 'Plot distribution of <column>')."
    )
    return state


def request_clarification_node(state: AgentState) -> AgentState:
    """Asks user for clarification instead of running ambiguous code."""
    state["generated_code"] = ""
    state["execution_result"] = {"status": "success", "output": "", "figure": None, "plotly_figure": None}
    state["needs_visualization"] = False
    state["off_topic"] = False
    state["intent"] = "CLARIFICATION"
    state["needs_clarification"] = True

    if not state.get("clarification_message"):
        msg, questions = generate_clarification_for_ambiguity(
            state.get("query", ""),
            state.get("columns", []),
            state.get("numeric_columns", []),
            state.get("categorical_columns", [])
        )
        state["clarification_message"] = msg
        state["suggested_questions"] = questions

    state["insights"] = state.get("clarification_message") or "Please specify which columns you would like to analyze."
    return state


def conversational_response_node(state: AgentState) -> AgentState:
    """Handles conversational follow-ups and explains prior insights or doubts in depth without code."""
    llm = get_llm(state)
    sys_prompt = f"""You are a Lead AI Data Analyst Assistant in Chat Mode.
The user is a business stakeholder asking follow-up questions to discuss previous insights, clarify doubts, or explore implications.

DATASET CONTEXT & SCHEMA:
{state.get('dataset_info', '')}

GUIDELINES:
1. Provide a thorough, articulate, and crystal-clear explanation addressing the user's specific doubts or follow-up question.
2. Reference the dataset columns, metrics, and previously observed patterns from the chat history.
3. Explain the business meaning, domain context, and causal factors behind the data findings.
4. Do not output or write Python code. Communicate clearly, professionally, and helpfully."""

    messages = [("system", sys_prompt)]
    for msg in state.get("chat_history", []):
        messages.append((msg.get("role", "user"), msg.get("content", "")))
    messages.append(("user", state["query"]))

    result = llm.invoke(messages)
    state["insights"] = result.content
    state["generated_code"] = ""
    state["execution_result"] = {"status": "success", "output": "", "figure": None, "plotly_figure": None}
    return state


def generate_code_node(state: AgentState) -> AgentState:
    """Generates pandas/plotly/seaborn Python code based on query and dataset schema."""
    llm = get_llm(state)
    needs_viz = state.get("needs_visualization", False)

    sys_prompt = f"""You are an expert Python data analyst.
The dataset is ALREADY loaded in variable `df`.
A numeric-only view is available in `numeric_df`.

RULES:
1. Use only pandas, numpy, matplotlib.pyplot as plt, seaborn as sns, and plotly.express as px.
2. DO NOT load the dataset. `df` is ALREADY available and loaded with data in memory. NEVER write `pd.read_csv()` or use dummy filenames like 'path_to_your_data.csv'.
3. If NEEDS_VISUALIZATION is TRUE:
   - For Plotly: assign figure to `plotly_fig` or `fig` (e.g. `fig = px.histogram(df, x='column')`). NEVER call `fig.show()`.
   - For a bar chart or distribution of a categorical column (e.g. 'sex', 'class', 'species'):
     Use frequency value counts:
     ```python
     counts = df['column'].value_counts().reset_index()
     counts.columns = ['column', 'count']
     fig = px.bar(counts, x='column', y='count', title='Distribution of column')
     ```
   - For Matplotlib/Seaborn: create the plot. NEVER call `plt.show()`.
   - Always calculate and print the key numerical summary findings (e.g. value counts, group means, or correlations) using `print()`.
4. If NEEDS_VISUALIZATION is FALSE:
   - **DO NOT GENERATE ANY PLOTS OR FIGURES.**
   - Calculate the requested statistics and print the results with `print()`.
5. For groupby aggregations: always use `numeric_only=True` (e.g. `df.groupby('sex').mean(numeric_only=True)`) or select the specific column (e.g. `df.groupby('sex')['fare'].mean()`).
6. For correlations or heatmaps:
   - Use `numeric_df.corr()` or `df.corr(numeric_only=True)`.
7. Return ONLY executable Python code inside ```python ``` blocks. No markdown explanations outside."""

    user_content = (
        f"DATASET INFO:\n{state['dataset_info']}\n\n"
        f"USER QUERY: {state['query']}\n\n"
        f"NEEDS VISUALIZATION: {needs_viz}"
    )

    messages = [
        ("system", sys_prompt),
        ("user", user_content)
    ]

    result = llm.invoke(messages)
    code_text = result.content
    match = re.search(r"```python\s*(.*?)\s*```", code_text, re.DOTALL)
    if match:
        code_text = match.group(1)

    state["generated_code"] = code_text.strip()
    return state


def execute_code_node(state: AgentState) -> AgentState:
    """Executes code via ExecutionEngine and guarantees figure suppression if no viz requested."""
    try:
        dataset_path = state["dataset_path"]
        df, _ = load_dataset(dataset_path, use_cache=True)
        code = state["generated_code"]
        query = state.get("query", "")

        result = execution_engine.execute_code(code, df, query)

        # Fail-safe guardrail: If the query did NOT request a visualization, suppress any generated figure!
        if not state.get("needs_visualization", False):
            result["figure"] = None
            result["plotly_figure"] = None

        state["execution_result"] = result
    except Exception as e:
        state["execution_result"] = {
            "status": "error",
            "error": str(e),
            "output": "",
            "figure": None,
            "plotly_figure": None,
        }
    return state


def generate_insights_node(state: AgentState) -> AgentState:
    """Generates natural language analytical insights tailored for executive stakeholders (3-5 lines max)."""
    exec_res = state["execution_result"]
    if exec_res.get("status") == "error":
        state["insights"] = f"An error occurred during analysis: {exec_res.get('error', 'Unknown execution error')}"
        return state

    llm = get_llm(state)
    sys_prompt = """You are a Lead AI Data Analyst presenting key findings directly to an Executive Stakeholder.

RULES FOR INSIGHTS:
1. Speak like a Lead Analyst presenting to leadership. Be sharp, professional, and data-driven.
2. Focus strictly on WHAT THE DATA REVEALS (the key business/domain findings, strongest trends, significant correlations, or major anomalies).
3. DO NOT describe generic chart mechanics (e.g. NEVER say "This heatmap has a colorbar", "The color scale is 1 to 1", "The x-axis displays...", "This plot shows..."). Explain the ACTUAL FINDINGS from the numbers!
4. LENGTH: Keep it to a SHORT SUMMARY in 3 to 5 lines maximum. Format as 2-3 crisp bullet points with exact numbers.
5. End with a helpful one-line note:
   "💡 *Want to explore these findings further? Switch to **💬 Chat (Discuss Insights)** mode to ask follow-up questions.*"
"""

    user_msg_content = (
        f"USER QUERY: {state['query']}\n\n"
        f"EXECUTED PYTHON CODE:\n{state.get('generated_code','')}\n\n"
        f"DATA FINDINGS & STATISTICAL OUTPUT:\n{exec_res.get('output','')}\n\n"
        "Provide your executive stakeholder summary (3-5 lines max)."
    )

    messages = [
        ("system", sys_prompt),
        ("user", user_msg_content)
    ]

    result = llm.invoke(messages)
    state["insights"] = result.content
    return state


def build_workflow():
    workflow = StateGraph(AgentState)

    # Nodes
    workflow.add_node("inspect_data", inspect_data_node)
    workflow.add_node("generate_suggestions", generate_suggestions_node)
    workflow.add_node("understand_query", understand_query_node)
    workflow.add_node("handle_meta_questions", handle_meta_questions_node)
    workflow.add_node("reject_off_topic", reject_off_topic_node)
    workflow.add_node("request_clarification", request_clarification_node)
    workflow.add_node("conversational_response", conversational_response_node)
    workflow.add_node("generate_code", generate_code_node)
    workflow.add_node("execute_code", execute_code_node)
    workflow.add_node("generate_insights", generate_insights_node)

    # Entry point & Initial routing
    def route_query_initial(state: AgentState):
        if not state.get("query", "").strip() and state.get("chat_mode") != "chat":
            return "generate_suggestions"
        if state.get("chat_mode") == "chat":
            return "conversational_response"
        return "understand_query"

    workflow.set_entry_point("inspect_data")
    workflow.add_conditional_edges("inspect_data", route_query_initial, {
        "generate_suggestions": "generate_suggestions",
        "conversational_response": "conversational_response",
        "understand_query": "understand_query",
    })

    workflow.add_edge("generate_suggestions", END)

    # Routing after understanding query
    def route_after_understand(state: AgentState):
        if state.get("off_topic") or state.get("intent") == "OFF_TOPIC":
            return "reject_off_topic"
        if state.get("needs_clarification") or state.get("intent") == "CLARIFICATION":
            return "request_clarification"
        if state.get("intent") == "META_QUESTIONS":
            return "handle_meta_questions"
        if state.get("intent") == "EXPLANATION":
            return "conversational_response"
        return "generate_code"

    workflow.add_conditional_edges("understand_query", route_after_understand, {
        "reject_off_topic": "reject_off_topic",
        "request_clarification": "request_clarification",
        "handle_meta_questions": "handle_meta_questions",
        "conversational_response": "conversational_response",
        "generate_code": "generate_code",
    })

    workflow.add_edge("handle_meta_questions", END)
    workflow.add_edge("conversational_response", END)
    workflow.add_edge("reject_off_topic", END)
    workflow.add_edge("request_clarification", END)
    workflow.add_edge("generate_code", "execute_code")
    workflow.add_edge("execute_code", "generate_insights")
    workflow.add_edge("generate_insights", END)

    return workflow.compile()


app = build_workflow()
