import sys
from pathlib import Path

# Allow `python backend/main.py` and IDE "run file" when cwd is wrong: package root must be on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Load .env from project root
from dotenv import load_dotenv
load_dotenv(_PROJECT_ROOT / ".env")

import json
import logging
import os
import shutil
import uuid
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import requests as http_requests

from backend.agents.workflow import app as agent_workflow, get_llm
from backend.universal_loader import load_dataset, get_dataset_profiling_snippet
from backend.pipelines.legacy.preprocess import preprocess_dataset
from backend.pipelines.core.enhanced_preprocess import preprocess_dataset_enhanced
from backend.pipelines.core.enhanced_data_cleaning import clean_dataset_enhanced

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

SUPPORTED_EXTENSIONS = (
    ".csv", ".tsv", ".xlsx", ".xls", ".json", ".jsonl", ".ndjson", ".parquet", ".pq", ".txt"
)


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


def _save_dataframe_as(df: pd.DataFrame, target_path: str, ext: str) -> None:
    """Saves DataFrame into matching target format."""
    ext = ext.lower()
    if ext in [".xlsx", ".xls"]:
        df.to_excel(target_path, index=False)
    elif ext in [".parquet", ".pq"]:
        df.to_parquet(target_path, index=False)
    elif ext in [".json", ".jsonl"]:
        df.to_json(target_path, orient="records", lines=ext.endswith("l"))
    else:
        df.to_csv(target_path, index=False)


@app.get("/")
async def root():
    return {"status": "ok", "message": "AI Data Analyst Agent API is running"}


@app.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """Uploads any supported dataset format and runs cleaning pipeline."""
    ext = os.path.splitext(file.filename or "")[1].lower()
    if not file.filename or not file.filename.lower().endswith(SUPPORTED_EXTENSIONS):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format. Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    dataset_id = str(uuid.uuid4())
    original_filename = f"{dataset_id}_original{ext}"
    cleaned_filename = f"{dataset_id}_cleaned{ext}"
    save_path = os.path.join(DATASETS_DIR, original_filename)

    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        df, meta = load_dataset(save_path, use_cache=True)
        cleaning_result = preprocess_dataset(df)
        cleaned_df: pd.DataFrame = cleaning_result["cleaned_df"]
        cleaning_summary: str = cleaning_result["cleaning_summary"]
        recommended_features = cleaning_result.get("recommended_features", [])

        is_dirty = not cleaned_df.equals(df)
        cleaned_id = None
        if is_dirty:
            cleaned_save_path = os.path.join(DATASETS_DIR, cleaned_filename)
            _save_dataframe_as(cleaned_df, cleaned_save_path, ext)
            cleaned_id = cleaned_filename

        return {
            "dataset_id": original_filename,
            "filename": file.filename,
            "is_dirty": is_dirty,
            "original_id": original_filename,
            "cleaned_id": cleaned_id,
            "cleaning_summary": cleaning_summary,
            "recommended_features": recommended_features,
            "meta": meta,
        }
    except Exception as e:
        logger.error(f"Error in upload data cleaning: {e}", exc_info=True)
        return {
            "dataset_id": original_filename,
            "filename": file.filename,
            "is_dirty": False,
            "original_id": original_filename,
            "cleaned_id": None,
            "cleaning_summary": f"Data loaded without auto-cleaning due to: {e}",
            "recommended_features": [],
        }


@app.post("/upload-enhanced")
async def upload_dataset_enhanced(
    file: UploadFile = File(...),
    use_enhanced: bool = True,
    preprocessing_config: str = "{}",
    cleaning_config: str = "{}"
):
    """Enhanced multi-format upload endpoint with quality scoring and advanced cleaning."""
    ext = os.path.splitext(file.filename or "")[1].lower()
    if not file.filename or not file.filename.lower().endswith(SUPPORTED_EXTENSIONS):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format. Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    try:
        preprocess_config = json.loads(preprocessing_config) if preprocessing_config else {}
        clean_config = json.loads(cleaning_config) if cleaning_config else {}
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON in configuration parameters")

    dataset_id = str(uuid.uuid4())
    original_filename = f"{dataset_id}_original{ext}"
    cleaned_filename = f"{dataset_id}_cleaned{ext}"
    save_path = os.path.join(DATASETS_DIR, original_filename)

    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        df, meta = load_dataset(save_path, use_cache=True)

        if use_enhanced:
            cleaning_result = clean_dataset_enhanced(df, clean_config)
            cleaned_df = cleaning_result["cleaned_df"]
            cleaning_summary = cleaning_result["cleaning_summary"]
            data_quality_score = cleaning_result.get("data_quality_score")

            preprocessing_result = preprocess_dataset_enhanced(cleaned_df, preprocess_config)
            recommended_features = preprocessing_result["recommended_features"]
            combined_summary = f"ENHANCED CLEANING:\n{cleaning_summary}\n\nENHANCED PREPROCESSING:\n{preprocessing_result['cleaning_summary']}"

            metadata = {
                "cleaning_report": cleaning_result.get("detailed_steps", {}),
                "data_quality_score": data_quality_score,
                "pipeline_version": "enhanced_v2.0",
                "loader_meta": meta,
            }
        else:
            cleaning_result = preprocess_dataset(df)
            cleaned_df = cleaning_result["cleaned_df"]
            combined_summary = cleaning_result["cleaning_summary"]
            recommended_features = cleaning_result.get("recommended_features", [])
            data_quality_score = None
            metadata = {"pipeline_version": "original", "loader_meta": meta}

        is_dirty = not cleaned_df.equals(df)
        cleaned_id = None
        if is_dirty:
            cleaned_save_path = os.path.join(DATASETS_DIR, cleaned_filename)
            _save_dataframe_as(cleaned_df, cleaned_save_path, ext)
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

        if use_enhanced and data_quality_score:
            response["data_quality_score"] = data_quality_score

        return response
    except Exception as e:
        logger.error(f"Enhanced data processing failed: {e}", exc_info=True)
        return {
            "dataset_id": original_filename,
            "filename": file.filename,
            "is_dirty": False,
            "original_id": original_filename,
            "cleaned_id": None,
            "cleaning_summary": f"Data loaded without auto-cleaning: {e}",
            "recommended_features": [],
            "pipeline_used": "enhanced",
            "error": str(e),
        }


@app.post("/ask")
async def ask_question(request: QueryRequest):
    """Passes user query to LangGraph agent with cached dataset loader and guardrails."""
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
        final_state = agent_workflow.invoke(initial_state)

        return {
            "query": final_state.get("query", ""),
            "dataset_info_snippet": final_state.get("dataset_info", ""),
            "insights": final_state.get("insights", ""),
            "figure": final_state.get("execution_result", {}).get("figure"),
            "plotly_figure": final_state.get("execution_result", {}).get("plotly_figure"),
            "code": final_state.get("generated_code", ""),
            "suggested_questions": final_state.get("suggested_questions", []),
            "intent": final_state.get("intent", ""),
            "needs_visualization": final_state.get("needs_visualization", False),
            "off_topic": final_state.get("off_topic", False),
        }
    except Exception as e:
        logger.error(f"Error in ask_question: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/download/{dataset_id}")
async def download_dataset(dataset_id: str):
    """Download a specific dataset file."""
    file_path = os.path.join(DATASETS_DIR, dataset_id)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    media_type = "text/csv"
    if dataset_id.endswith(".xlsx"):
        media_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    elif dataset_id.endswith(".parquet") or dataset_id.endswith(".pq"):
        media_type = "application/octet-stream"
    elif dataset_id.endswith(".json"):
        media_type = "application/json"

    return FileResponse(path=file_path, filename=dataset_id, media_type=media_type)


@app.get("/model-ready/{dataset_id}")
async def get_model_ready_dataset(dataset_id: str):
    """Build and return a model-ready dataset using universal loading."""
    file_path = os.path.join(DATASETS_DIR, dataset_id)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Dataset not found")

    df, _ = load_dataset(file_path, use_cache=True)
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
    """Streaming conversational endpoint for follow-up discussions and stakeholder Q&A."""
    dataset_path = os.path.join(DATASETS_DIR, request.dataset_id)
    if not os.path.exists(dataset_path):
        raise HTTPException(status_code=404, detail="Dataset not found")

    df, _ = load_dataset(dataset_path, use_cache=True)
    dataset_info = get_dataset_profiling_snippet(df)

    state = {
        "provider": request.provider,
        "model_name": request.model_name,
        "api_key": request.api_key,
        "ollama_base_url": request.ollama_base_url,
    }
    llm = get_llm(state)

    sys_prompt = f"""You are a Lead AI Data Analyst Assistant communicating directly with a business stakeholder in Chat Mode.
The stakeholder is asking follow-up questions to discuss previous findings, clarify doubts, understand trends, or explore next steps.

DATASET SCHEMA & SUMMARY:
{dataset_info}

GUIDELINES:
1. Provide a clear, thorough, and highly insightful response directly addressing the user's specific doubts or follow-up question.
2. Reference the dataset columns, metrics, and previously observed patterns from the conversation history.
3. Explain the business meaning and real-world implications behind the data findings with clarity and depth.
4. Do not output Python code. Communicate in a consultative, professional, and accessible executive tone."""
    messages = [("system", sys_prompt)]

    for msg in request.chat_history:
        messages.append((msg.get("role", "user"), msg.get("content", "")))

    messages.append(("user", request.query))

    def token_generator():
        for chunk in llm.stream(messages):
            content = getattr(chunk, "content", "") or ""
            if not content:
                continue
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
    """Checks if Ollama instance is reachable."""
    url = base_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    native_url = url.rstrip("/")
    if native_url.endswith("/v1"):
        native_url = native_url[:-3]

    try:
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
