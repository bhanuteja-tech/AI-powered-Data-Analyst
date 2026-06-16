import os
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from backend.agents.state import AgentState
from backend.execution_engine import ExecutionEngine
import re

execution_engine = ExecutionEngine()

def get_llm(state: AgentState):
    provider = state.get("provider", "openai")
    model = state.get("model_name", "gpt-4o-mini")
    api_key = state.get("api_key", "")
    
    if provider == "openrouter":
        return ChatOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key or os.environ.get("OPENROUTER_API_KEY", "dummy"),
            model=model,
            temperature=0
        )
    elif provider == "ollama":
        # Ollama instance - configurable base URL for production support
        ollama_base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        return ChatOpenAI(
            base_url=ollama_base_url,
            api_key="ollama", # required but dummy
            model=model,
            temperature=0
        )
    else: # openai
        return ChatOpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY", "dummy"),
            model=model,
            temperature=0
        )

def inspect_data_node(state: AgentState) -> AgentState:
    """Analyzes the dataset and extracts schema and summary info."""
    dataset_path = state["dataset_path"]
    _, ext = os.path.splitext(dataset_path.lower())
    if ext == ".xlsx":
        df = pd.read_excel(dataset_path)
    else:
        df = pd.read_csv(dataset_path)
    
    buffer = []
    buffer.append(f"Shape: {df.shape}")
    buffer.append("\nColumns and Data Types:")
    buffer.append(df.dtypes.to_string())
    
    buffer.append("\nMissing Values:")
    buffer.append(df.isnull().sum().to_string())
    
    buffer.append("\nSummary Statistics:")
    buffer.append(df.describe(include='all').to_string())
    
    # Get a sample of the first 3 rows
    buffer.append("\nSample Data (First 3 rows):")
    buffer.append(df.head(3).to_string())
    
    state["dataset_info"] = "\n".join(buffer)
    state["columns"] = df.columns.tolist()
    state["numeric_columns"] = df.select_dtypes(include=["number"]).columns.tolist()
    state["categorical_columns"] = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    return state

def generate_suggestions_node(state: AgentState) -> AgentState:
    """Generates 3 suggested questions based on dataset schema, if query is empty."""
    llm = get_llm(state)
    messages = [
        ("system", "You are an AI data assistant. Based on the following dataset schema and summary, suggest exactly 3 interesting analytical questions a user could ask. Return ONLY the 3 questions as a bulleted list, one per line. Do not include introductory text."),
        ("user", f"DATASET INFO:\n{state['dataset_info']}")
    ]
    
    result = llm.invoke(messages)
    
    # Parse bullet points into a list
    questions = []
    for line in result.content.split('\n'):
        line = line.strip()
        if line.startswith('- ') or line.startswith('* '):
            questions.append(line[2:])
        elif re.match(r'^\d+\.\s', line):
            questions.append(re.sub(r'^\d+\.\s*', '', line))
        elif line:
            questions.append(line)
            
    state["suggested_questions"] = questions[:3]
    return state

def understand_query_node(state: AgentState) -> AgentState:
    """Analyzes user intent to determine if it's a data request vs off-topic."""
    llm = get_llm(state)
    
    state["needs_clarification"] = False
    state["clarification_message"] = ""

    # Deterministic guard: follow-ups referencing an existing visualization should be handled in Chat mode
    q = (state.get("query") or "").strip().lower()
    followup_markers = [
        "this graph", "this chart", "this plot", "this figure",
        "that graph", "that chart", "that plot", "that figure",
        "above graph", "above chart", "above plot", "above figure",
        "the graph above", "the chart above", "the plot above",
        "previous graph", "previous chart", "previous plot", "previous figure",
        "earlier graph", "earlier chart", "earlier plot", "earlier figure",
        "what does this indicate", "what does this mean", "explain this graph",
        "interpret this graph", "interpret this chart", "interpret this plot",
    ]
    if any(m in q for m in followup_markers) or re.search(r"\b(this|that|above|previous|earlier)\b.*\b(graph|chart|plot|figure|visual)\b", q):
        state["intent"] = "OFF_TOPIC"
        state["needs_visualization"] = False
        state["off_topic"] = True
        return state

    # Strict mode: underspecified visualization requests -> ask clarification, do not execute
    viz_markers = ["plot", "chart", "graph", "visual", "visualize", "heatmap", "scatter", "bar", "line", "hist", "box", "violin", "facet", "facetgrid", "facet grid", "pairplot"]
    if any(m in q for m in viz_markers):
        cols = state.get("columns", []) or []
        # if query doesn't mention any known column, it’s likely underspecified
        mentions_col = any(c.lower() in q for c in cols if isinstance(c, str) and c)
        wants_facet = ("facet" in q) or ("facetgrid" in q) or ("facet grid" in q)
        if wants_facet and not mentions_col:
            numeric_cols = state.get("numeric_columns", []) or []
            cat_cols = state.get("categorical_columns", []) or []

            default_group = cat_cols[0] if cat_cols else None
            numeric_preview = ", ".join(numeric_cols[:8]) if numeric_cols else "(none detected)"
            cat_preview = ", ".join(cat_cols[:8]) if cat_cols else "(none detected)"

            suggestions = []
            if default_group and numeric_cols:
                suggestions.append(f"- Facet histograms of `{numeric_cols[0]}` split by `{default_group}`")
                suggestions.append(f"- Facet scatter: `{numeric_cols[0]}` vs `{numeric_cols[1]}` split by `{default_group}`" if len(numeric_cols) > 1 else "")
            if numeric_cols:
                suggestions.append(f"- Facet distributions for all numeric columns (melted) by `{default_group}`" if default_group else "- Facet distributions for all numeric columns (melted)")
            suggestions = [s for s in suggestions if s]

            state["needs_clarification"] = True
            state["clarification_message"] = (
                "Your request is a bit underspecified for a facet grid (which variables to plot and how to split/facet). "
                "I won’t run a best‑guess chart in **Analysis mode**.\n\n"
                f"**Available numeric columns:** {numeric_preview}\n\n"
                f"**Available categorical columns (good for faceting):** {cat_preview}\n\n"
                "**Pick one option (or tell me X/Y + facet column):**\n"
                + ("\n".join(suggestions) if suggestions else "- Tell me which numeric columns to plot and which categorical column to facet by.")
            )
            return state

    # Format chat history for context
    chat_context = ""
    history = state.get("chat_history", [])
    if history:
        chat_context = "RECENT CHAT HISTORY:\n"
        # Take the last 3 exchanges to avoid context bloat
        for msg in history[-6:]:
            chat_context += f"{msg['role'].upper()}: {msg['content']}\n"
    
    sys_prompt = f"""You are an AI assistant analyzing a user's prompt in a dataset exploration tool.
Determine the user's INTENT based on the current prompt and the chat history.

OPTIONS:
1. "VISUALIZATION": User explicitly asks for a graph, plot, chart, or trend analysis that requires coding a visual.
2. "DATA_ONLY": User asks a quantitative question requiring pandas code to calculate (e.g., max, min, average, sum) but no chart.
3. "OFF_TOPIC": User asks for something not related to analyzing the dataset (e.g., "give me Fibonacci code", general programming help, generic chat, writing emails, etc.). This should NOT run code on the dataset.

{chat_context}

Respond with exactly ONE word: VISUALIZATION or DATA_ONLY or OFF_TOPIC."""

    messages = [
        ("system", sys_prompt),
        ("user", state["query"])
    ]
    
    result = llm.invoke(messages)
    
    intent_str = result.content.strip().upper()
    state["intent"] = intent_str
    state["needs_visualization"] = "VISUALIZATION" in intent_str
    state["off_topic"] = "OFF_TOPIC" in intent_str
    return state


def reject_off_topic_node(state: AgentState) -> AgentState:
    """Rejects non-dataset queries in analysis mode."""
    state["generated_code"] = ""
    state["execution_result"] = {}
    state["insights"] = (
        "This question doesn’t look like a dataset analysis request for the currently loaded data. "
        "Please switch to **💬 Chat (Discuss Insights)** for general questions, or ask a data-specific "
        "question like: “Show correlations”, “Plot X vs Y”, “What is the average of <column>?”, etc."
    )
    return state


def request_clarification_node(state: AgentState) -> AgentState:
    """Ask for clarification instead of executing underspecified analysis."""
    state["generated_code"] = ""
    state["execution_result"] = {}
    state["insights"] = state.get("clarification_message") or (
        "Please clarify which columns you want to plot and how you want to facet/group the visualization."
    )
    return state

def generate_code_node(state: AgentState) -> AgentState:
    """Generates the pandas/matplotlib/plotly code based on query and dataset info."""
    llm = get_llm(state)
    sys_prompt = """You are an expert Python data analyst. 
You are given the structure of a pandas dataframe `df`.
Your task is to write valid Python code to answer the user's query.
The dataset is ALREADY loaded in the variable `df`.

Wait, here are the RULES:
1. ONLY use pandas, numpy, matplotlib.pyplot as plt, seaborn as sns, and plotly.express as px.
2. DO NOT load the dataset. The variable `df` is already available.
3. If the user asks for a visualization:
   - If using Plotly (`px`), assign the final Figure to `plotly_fig`. DO NOT call `show()`. MUST ALWAYS use `.reset_index()` on grouped DataFrames beforehand so columns map correctly!
   - If using matplotlib/seaborn, just create the plot. DO NOT call `show()`.
   - You MUST ALSO use `print()` to output a statistical summary that describes the graph so the insight generator can understand it.
   - For scatter plots: you MUST print correlation (Pearson) and a small grouped summary if a categorical hue is used.
   - For bar/line charts: you MUST print the aggregated table you plotted.
   - For hist/box/violin: you MUST print describe() + skew() for the plotted numeric column(s).
4. If the user asks a factual question, or asks for data distribution/statistics (e.g., "what is the distribution?", "is it skewed?"):
   - You MUST print the statistical results using `print()`. 
   - For distributions, print skewness (`df.skew()`), kurtosis, or descriptive stats so the next parsing step can read it.
   - Do NOT just generate a plot if they ask for statistical text properties.
5. CRITICAL: For correlation analysis or any numeric operations:
   - ALWAYS filter to numeric columns first using: `numeric_df = df.select_dtypes(include=[np.number])`
   - Then calculate correlations on `numeric_df` only, NOT on the original `df`
   - This prevents errors when categorical columns (like 'S', 'C', 'Q') are present
   - Example: `correlation_matrix = numeric_df.corr()` instead of `df.corr()`
6. Return ONLY Python code inside ```python ``` blocks. Do not add explanations.
"""

    user_content = f"DATASET INFO:\n{state['dataset_info']}\n\nUSER QUERY: {state['query']}\n\nNEEDS VISUALIZATION: {state['needs_visualization']}"
    
    messages = [
        ("system", sys_prompt),
        ("user", user_content)
    ]
    
    result = llm.invoke(messages)
    
    # Extract code from markdown blocks
    code_text = result.content
    match = re.search(r'```python\n(.*?)\n```', code_text, re.DOTALL)
    if match:
        code_text = match.group(1)
        
    state["generated_code"] = code_text
    return state
    
def conversational_response_node(state: AgentState) -> AgentState:
    """Handles conversational follow-ups without running code."""
    llm = get_llm(state)
    
    sys_prompt = "You are an AI Data Analyst Assistant. Answer the user's conversational follow-up question based on the dataset info and conversation history. Do not write code. Be helpful and clear.\n\n"
    sys_prompt += "IMPORTANT: You cannot physically 'see' the visual graphs on the user's screen. However, you CAN see the exact Python code that was used to generate them in the chat history. Use that code to deduce what the graph looks like (what the axes are, what data is plotted) to answer the user's visual questions definitively.\n\n"
    sys_prompt += "DATASET INFO:\n" + state.get("dataset_info", "")
    
    messages = [
        ("system", sys_prompt)
    ]
    
    for msg in state.get("chat_history", []):
        messages.append((msg["role"], msg["content"]))
        
    messages.append(("user", state["query"]))
    
    # Send directly to LLM without PromptTemplate
    result = llm.invoke(messages)
    
    state["insights"] = result.content
    state["generated_code"] = ""
    state["execution_result"] = {}
    return state

def execute_code_node(state: AgentState) -> AgentState:
    """Executes the generated code securely and captures outputs."""
    try:
        dataset_path = state["dataset_path"]
        _, ext = os.path.splitext(dataset_path.lower())
        if ext == ".xlsx":
            df = pd.read_excel(dataset_path)
        else:
            df = pd.read_csv(dataset_path)
        code = state["generated_code"]
        
        # Execute using our engine
        result = execution_engine.execute_code(code, df)
        state["execution_result"] = result
        
    except Exception as e:
        state["execution_result"] = {
            "status": "error",
            "error": str(e),
            "output": "",
            "figure": None
        }
        
    return state

def generate_insights_node(state: AgentState) -> AgentState:
    """Generates natural language business insights from the execution results."""
    exec_res = state["execution_result"]
    if exec_res["status"] == "error":
        state["insights"] = f"An error occurred during analysis: {exec_res['error']}"
        return state
        
    llm = get_llm(state)
    sys_prompt = """You are a senior Data Analyst.
You must provide a clear, concise answer and business insight based on:
1) the user's query
2) the dataset metadata
3) the exact Python code that was executed
4) the printed output from that execution

IMPORTANT:
- Do NOT ask the user to "share the dataset" or "provide the data output" — the system already has it.
- If the printed output is sparse, infer what the chart represents from the executed code and still provide a best-effort interpretation.
- Keep it under 3-4 sentences. Be direct and helpful."""

    # Instead of template variables which cause problems with JSON/Dict outputs containing {},
    # we inject the output directly and pass raw messages to skip ChatPromptTemplate validation.
    user_msg_content = (
        f"USER QUERY: {state['query']}\n\n"
        f"DATASET METADATA (snippet):\n{state.get('dataset_info','')}\n\n"
        f"EXECUTED PYTHON CODE:\n{state.get('generated_code','')}\n\n"
        f"ANALYSIS TEXT OUTPUT (stdout/stderr):\n{exec_res.get('output','')}\n\n"
        "Provide the final answer to the user."
    )

    messages = [
        ("system", sys_prompt),
        ("user", user_msg_content)
    ]
    
    result = llm.invoke(messages)
    
    state["insights"] = result.content
    return state

# Define the Graph
def build_workflow():
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("inspect_data", inspect_data_node)
    workflow.add_node("generate_suggestions", generate_suggestions_node)
    workflow.add_node("understand_query", understand_query_node)
    workflow.add_node("reject_off_topic", reject_off_topic_node)
    workflow.add_node("request_clarification", request_clarification_node)
    workflow.add_node("conversational_response", conversational_response_node)
    workflow.add_node("generate_code", generate_code_node)
    workflow.add_node("execute_code", execute_code_node)
    workflow.add_node("generate_insights", generate_insights_node)
    
    # Conditional edge
    def route_query_initial(state: AgentState):
        if state.get("query", "").strip() == "" and state.get("chat_mode") != "chat":
            return "generate_suggestions"
        if state.get("chat_mode") == "chat":
            return "conversational_response"
        return "understand_query"

    # Edges
    workflow.set_entry_point("inspect_data")
    workflow.add_conditional_edges("inspect_data", route_query_initial, {
        "generate_suggestions": "generate_suggestions",
        "conversational_response": "conversational_response",
        "understand_query": "understand_query"
    })
    
    workflow.add_edge("generate_suggestions", END)
    def route_after_understand(state: AgentState):
        if state.get("off_topic"):
            return "reject_off_topic"
        if state.get("needs_clarification"):
            return "request_clarification"
        return "generate_code"

    workflow.add_conditional_edges("understand_query", route_after_understand, {
        "reject_off_topic": "reject_off_topic",
        "request_clarification": "request_clarification",
        "generate_code": "generate_code",
    })
    workflow.add_edge("conversational_response", END)
    workflow.add_edge("reject_off_topic", END)
    workflow.add_edge("request_clarification", END)
    workflow.add_edge("generate_code", "execute_code")
    workflow.add_edge("execute_code", "generate_insights")
    workflow.add_edge("generate_insights", END)
    
    return workflow.compile()

app = build_workflow()
