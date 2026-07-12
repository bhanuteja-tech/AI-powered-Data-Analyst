import sys
from pathlib import Path

# Allow `python backend/main.py` and IDE "run file" when cwd is wrong: package root must be on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Load .env from project root
from dotenv import load_dotenv
load_dotenv(_PROJECT_ROOT / ".env")

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import shutil
import uuid
import pandas as pd
import requests as http_requests  # renamed to avoid conflict with FastAPI Request
from backend.agents.workflow import app as agent_workflow, get_llm
from backend.pipelines.legacy.preprocess import preprocess_dataset
from backend.pipelines.core.enhanced_preprocess import preprocess_dataset_enhanced
from backend.pipelines.core.enhanced_data_cleaning import clean_dataset_enhanced

app = FastAPI(title="AI Data Analyst Agent API")

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATASETS_DIR = os.path.join(os.path.dirname(__file__), "..", "datasets")
os.makedirs(DATASETS_DIR, exist_ok=True)

class QueryRequest(BaseModel):
    dataset_id: str
    query: str
    provider: str = "openai"
    model_name: str = "gpt-4o-mini"
    api_key: str = ""
    chat_history: list[dict] = []
    chat_mode: str = "analysis"
    ollama_base_url: str = ""


class StreamQueryRequest(BaseModel):
    dataset_id: str
    query: str
    provider: str = "openai"
    model_name: str = "gpt-4o-mini"
    api_key: str = ""
    chat_history: list[dict] = []
    ollama_base_url: str = ""


class EnhancedUploadRequest(BaseModel):
    use_enhanced_pipeline: bool = True
    preprocessing_config: dict = {}
    cleaning_config: dict = {}

@app.get("/")
async def root():
    return {"status": "ok", "message": "AI Data Analyst Agent API is running"}

@app.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """Uploads a dataset (CSV) and returns a dataset UUID."""
    if not file.filename.endswith(('.csv', '.xlsx')):
        raise HTTPException(status_code=400, detail="Only CSV or Excel files are supported")
    
    dataset_id = str(uuid.uuid4())
    ext = os.path.splitext(file.filename)[1]
    original_filename = f"{dataset_id}_original{ext}"
    cleaned_filename = f"{dataset_id}_cleaned{ext}"
    save_path = os.path.join(DATASETS_DIR, original_filename)
    
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # Read the dataset
        if ext == '.csv':
            df = pd.read_csv(save_path)
        else:
            df = pd.read_excel(save_path)

        # Run full professional cleaning pipeline
        cleaning_result = preprocess_dataset(df)
        cleaned_df: pd.DataFrame = cleaning_result["cleaned_df"]
        cleaning_summary: str = cleaning_result["cleaning_summary"]
        recommended_features = cleaning_result.get("recommended_features", [])

        # Determine if any changes were made by basic shape/summary comparison
        is_dirty = not cleaned_df.equals(df)

        cleaned_id = None
        if is_dirty:
            cleaned_save_path = os.path.join(DATASETS_DIR, cleaned_filename)
            if ext == '.csv':
                cleaned_df.to_csv(cleaned_save_path, index=False)
            else:
                cleaned_df.to_excel(cleaned_save_path, index=False)
            cleaned_id = cleaned_filename

        return {
            "dataset_id": original_filename,
            "filename": file.filename,
            "is_dirty": is_dirty,
            "original_id": original_filename,
            "cleaned_id": cleaned_id,
            "cleaning_summary": cleaning_summary,
            "recommended_features": recommended_features,
        }
    except Exception as e:
        error_text = f"Data cleaning failed due to an internal error: {e}"
        print(f"Error checking/cleaning dataset: {e}")
        return {
            "dataset_id": original_filename,
            "filename": file.filename,
            "is_dirty": False,
            "original_id": original_filename,
            "cleaned_id": None,
            "cleaning_summary": error_text,
            "recommended_features": [],
        }

@app.post("/upload-enhanced")
async def upload_dataset_enhanced(
    file: UploadFile = File(...), 
    use_enhanced: bool = True,
    preprocessing_config: str = "{}",
    cleaning_config: str = "{}"
):
    """Enhanced upload endpoint with advanced preprocessing and cleaning pipelines."""
    if not file.filename.endswith(('.csv', '.xlsx')):
        raise HTTPException(status_code=400, detail="Only CSV or Excel files are supported")
    
    try:
        # Parse configuration JSON strings
        import json
        preprocess_config = json.loads(preprocessing_config) if preprocessing_config else {}
        clean_config = json.loads(cleaning_config) if cleaning_config else {}
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON in configuration parameters")
    
    dataset_id = str(uuid.uuid4())
    ext = os.path.splitext(file.filename)[1]
    original_filename = f"{dataset_id}_original{ext}"
    cleaned_filename = f"{dataset_id}_cleaned{ext}"
    save_path = os.path.join(DATASETS_DIR, original_filename)
    
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # Read the dataset
        if ext == '.csv':
            df = pd.read_csv(save_path)
        else:
            df = pd.read_excel(save_path)

        # Use enhanced pipeline if requested
        if use_enhanced:
            # First clean the data
            cleaning_result = clean_dataset_enhanced(df, clean_config)
            cleaned_df = cleaning_result["cleaned_df"]
            cleaning_summary = cleaning_result["cleaning_summary"]
            data_quality_score = cleaning_result.get("data_quality_score")
            
            # Then preprocess for modeling
            preprocessing_result = preprocess_dataset_enhanced(cleaned_df, preprocess_config)
            model_ready_df = preprocessing_result["model_ready_df"]
            recommended_features = preprocessing_result["recommended_features"]
            
            # Combine reports
            combined_summary = f"ENHANCED CLEANING:\n{cleaning_summary}\n\nENHANCED PREPROCESSING:\n{preprocessing_result['cleaning_summary']}"
            
            metadata = {
                "cleaning_report": cleaning_result["detailed_steps"],
                "preprocessing_report": preprocessing_result["metadata"]["preprocessing_report"],
                "data_quality_score": data_quality_score,
                "pipeline_version": "enhanced_v2.0"
            }
        else:
            # Fall back to original pipeline
            cleaning_result = preprocess_dataset(df)
            cleaned_df = cleaning_result["cleaned_df"]
            cleaning_summary = cleaning_result["cleaning_summary"]
            recommended_features = cleaning_result.get("recommended_features", [])
            model_ready_df = cleaning_result.get("model_ready_df", cleaned_df)
            combined_summary = cleaning_summary
            metadata = {"pipeline_version": "original"}

        # Check if any changes were made
        is_dirty = not cleaned_df.equals(df)

        cleaned_id = None
        if is_dirty:
            cleaned_save_path = os.path.join(DATASETS_DIR, cleaned_filename)
            if ext == '.csv':
                cleaned_df.to_csv(cleaned_save_path, index=False)
            else:
                cleaned_df.to_excel(cleaned_save_path, index=False)
            cleaned_id = cleaned_filename

        response = {
            "dataset_id": original_filename,
            "filename": file.filename,
            "is_dirty": is_dirty,
            "original_id": original_filename,
            "cleaned_id": cleaned_id,
            "cleaning_summary": combined_summary,
            "recommended_features": recommended_features,
            "metadata": metadata,
            "pipeline_used": "enhanced" if use_enhanced else "original"
        }

        # Add data quality score if available
        if use_enhanced and data_quality_score:
            response["data_quality_score"] = data_quality_score

        return response

    except Exception as e:
        error_text = f"Enhanced data processing failed: {e}"
        print(f"Error in enhanced processing: {e}")
        return {
            "dataset_id": original_filename,
            "filename": file.filename,
            "is_dirty": False,
            "original_id": original_filename,
            "cleaned_id": None,
            "cleaning_summary": error_text,
            "recommended_features": [],
            "pipeline_used": "enhanced",
            "error": str(e)
        }

@app.post("/ask")
async def ask_question(request: QueryRequest):
    """Passes the query to the LangGraph agent for analysis."""
    dataset_path = os.path.join(DATASETS_DIR, request.dataset_id)
    if not os.path.exists(dataset_path):
        raise HTTPException(status_code=404, detail="Dataset not found")
        
    initial_state = {
        "query": request.query,
        "dataset_path": dataset_path,
        "provider": request.provider,
        "model_name": request.model_name,
        "api_key": request.api_key,
        "chat_history": request.chat_history,
        "chat_mode": request.chat_mode,
        "ollama_base_url": request.ollama_base_url,
    }
    
    try:
        # Run agent workflow
        final_state = agent_workflow.invoke(initial_state)
        
        return {
            "query": final_state["query"],
            "dataset_info_snippet": final_state.get("dataset_info", ""),
            "insights": final_state.get("insights", ""),
            "figure": final_state.get("execution_result", {}).get("figure"),
            "plotly_figure": final_state.get("execution_result", {}).get("plotly_figure"),
            "code": final_state.get("generated_code"),
            "suggested_questions": final_state.get("suggested_questions", [])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download/{dataset_id}")
async def download_dataset(dataset_id: str):
    """Download a specific dataset file."""
    file_path = os.path.join(DATASETS_DIR, dataset_id)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
        
    media_type = "text/csv" if dataset_id.endswith(".csv") else "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    return FileResponse(path=file_path, filename=dataset_id, media_type=media_type)


@app.get("/model-ready/{dataset_id}")
async def get_model_ready_dataset(dataset_id: str):
    """
    Build and return a model-ready dataset (cleaned, encoded, scaled) as CSV plus metadata.
    The caller can pass either the original or cleaned dataset_id.
    """
    file_path = os.path.join(DATASETS_DIR, dataset_id)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Dataset not found")

    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(file_path)
    else:
        df = pd.read_excel(file_path)

    result = preprocess_dataset(df)
    model_df: pd.DataFrame = result["model_ready_df"]
    metadata = result["metadata"]

    csv_str = model_df.to_csv(index=False)
    cleaned_df = result["cleaned_df"]
    cleaned_csv_str = cleaned_df.to_csv(index=False)

    return {
        "csv": csv_str,
        "columns": model_df.columns.tolist(),
        "cleaned_csv": cleaned_csv_str,
        "cleaned_columns": cleaned_df.columns.tolist(),
        "metadata": metadata,
    }


@app.post("/chat-stream")
def chat_stream(request: StreamQueryRequest):
    """
    Streaming conversational endpoint for chat mode.
    Streams the assistant's answer token-by-token, similar to ChatGPT/Claude.
    """
    dataset_path = os.path.join(DATASETS_DIR, request.dataset_id)
    if not os.path.exists(dataset_path):
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Build a minimal state-like dict for LLM configuration
    state = {
        "provider": request.provider,
        "model_name": request.model_name,
        "api_key": request.api_key,
        "ollama_base_url": request.ollama_base_url,
    }
    llm = get_llm(state)  # uses ChatOpenAI under the hood

    # System prompt similar to conversational_response_node
    sys_prompt = "You are an AI Data Analyst Assistant. Answer the user's conversational follow-up question based on the conversation history. Do not write code. Be helpful and clear.\n\n"
    messages = [("system", sys_prompt)]

    for msg in request.chat_history:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        messages.append((role, content))

    messages.append(("user", request.query))

    def token_generator():
        for chunk in llm.stream(messages):
            content = getattr(chunk, "content", "") or ""
            if not content:
                continue
            # LangChain messages may return a list of content parts
            if isinstance(content, list):
                text_parts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text_parts.append(part.get("text", ""))
                text = "".join(text_parts)
            else:
                text = str(content)
            if text:
                yield text

    return StreamingResponse(token_generator(), media_type="text/plain")


@app.get("/ollama-status")
async def ollama_status(base_url: str = ""):
    """
    Check if an Ollama instance is reachable.
    Returns the list of available models or an error message.
    """
    url = base_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    # Strip /v1 suffix to hit Ollama's native /api/tags endpoint
    native_url = url.rstrip("/")
    if native_url.endswith("/v1"):
        native_url = native_url[:-3]

    try:
        # ngrok free tier requires this header to skip the browser interstitial page
        headers = {"ngrok-skip-browser-warning": "true"}
        resp = http_requests.get(f"{native_url}/api/tags", timeout=5, headers=headers)
        if resp.status_code == 200:
            data = resp.json()
            models = [m.get("name", "unknown") for m in data.get("models", [])]
            return {"status": "connected", "models": models, "url": native_url}
        else:
            return {"status": "error", "detail": f"HTTP {resp.status_code}"}
    except Exception as e:
        return {"status": "error", "detail": str(e)}
