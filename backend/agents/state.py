from typing import TypedDict, Dict, Any, Optional

class AgentState(TypedDict):
    query: str
    dataset_path: str
    chat_history: list[dict[str, str]]
    chat_mode: str
    
    # LLM Config
    provider: str
    model_name: str
    api_key: str
    
    # Populated by Data Inspection Agent
    dataset_info: str
    columns: list[str]
    numeric_columns: list[str]
    categorical_columns: list[str]
    
    # Populated by Query Understanding Agent (Optional, could be an enhanced query or intent)
    intent: str
    
    # Populated by Code Generation Agent
    generated_code: str
    
    # Populated by Execution Agent (from ExecutionEngine)
    execution_result: Dict[str, Any]
    
    # Populated by Insight Agent
    insights: str
    
    # Any visualization specific details
    needs_visualization: bool

    # Guardrail: query is not a dataset analysis request (ask user to switch to Chat mode)
    off_topic: bool
    needs_clarification: bool
    clarification_message: str

    # Suggested questions based on schema (optional)
    suggested_questions: list[str]
