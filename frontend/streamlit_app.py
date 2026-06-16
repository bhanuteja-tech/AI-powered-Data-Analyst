import streamlit as st
import requests
import base64
import time
import plotly.io as pio
import os

# --- Configuration ---
API_BASE_URL = "https://ai-data-analyst-api-739z.onrender.com"

# Ollama model tags (ollama.com/library); pull with e.g. `ollama pull deepseek-v3.1:671b-cloud`
OLLAMA_MODEL_OPTIONS = [
    # Cloud (Ollama account / API)
    "deepseek-v3.1:671b-cloud",
    "deepseek-v3.2:cloud",
    "qwen3-coder:480b-cloud",
    "gpt-oss:120b-cloud",
    "gpt-oss:20b-cloud",
    # DeepSeek (local / other tags)
    "deepseek-r1:8b",
    "deepseek-r1:14b",
    "deepseek-r1:32b",
    "deepseek-r1:70b",
    "deepseek-coder-v2:16b",
    # Llama
    "llama3.3:70b",
    "llama3.2:3b",
    "llama3.2:1b",
    "llama3.1:8b",
    "llama3.1:70b",
    "llama3:8b",
    "llama3:70b",
    "llama3",
    # Qwen
    "qwen2.5:7b",
    "qwen2.5:14b",
    "qwen2.5:32b",
    "qwen2.5-coder:32b",
    "qwen2.5-coder:7b",
    # Mistral / Mixtral
    "mistral",
    "mistral-nemo",
    "mixtral:8x7b",
    "mixtral:8x22b",
    # Google Gemma
    "gemma2:9b",
    "gemma2:27b",
    "gemma:7b",
    # Microsoft / Meta small
    "phi4",
    "phi3",
    "codellama:7b",
    "codellama:13b",
]

st.set_page_config(
    page_title="AI Data Analyst Agent",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Premium UI Custom CSS ---
st.markdown("""
<style>
    /* Global Background and Fonts */
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
        font-family: 'Inter', sans-serif;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #FFFFFF !important;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    
    /* Main Title Styling */
    .main-title {
        background: linear-gradient(90deg, #A78BFA 0%, #F472B6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        color: #A0AEC0;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #171923;
        border-right: 1px solid #2D3748;
    }
    
    /* Chat Bubbles */
    .stChatMessage {
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    
    /* User Message */
    [data-testid="stChatMessage"][data-baseweb="card"]:has(div:contains("user")) {
        background: linear-gradient(135deg, #2D3748 0%, #1A202C 100%);
        border: 1px solid #4A5568;
    }

    /* Assistant Message */
    [data-testid="stChatMessage"][data-baseweb="card"]:has(div:contains("assistant")) {
        background-color: #1A202C;
        border: 1px solid #2D3748;
        border-left: 4px solid #A78BFA;
    }

    /* Info Boxes (Insights) */
    div.stInfo {
        background-color: rgba(167, 139, 250, 0.1) !important;
        border: 1px solid rgba(167, 139, 250, 0.3) !important;
        color: #E2E8F0 !important;
        border-radius: 8px;
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        background-color: #2D3748;
        border-radius: 8px;
        font-weight: 600;
    }

    /* Inputs */
    .stTextInput>div>div>input {
        background-color: #2D3748;
        color: white;
        border-radius: 8px;
        border: 1px solid #4A5568;
    }
    .stSelectbox>div>div>div {
        background-color: #2D3748;
        color: white;
        border-radius: 8px;
        border: 1px solid #4A5568;
    }
</style>
""", unsafe_allow_html=True)

# --- Session State ---
if "dataset_id" not in st.session_state:
    st.session_state.dataset_id = None
if "dataset_name" not in st.session_state:
    st.session_state.dataset_name = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Sidebar: User & Agent Config ---
with st.sidebar:
    st.markdown("### ⚙️ Engine Settings")
    llm_provider = st.selectbox("Provider", ["openai", "openrouter", "ollama"])
    
    if llm_provider == "openai":
        llm_model = st.selectbox("Model Name", ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"])
        api_key = st.text_input("OpenAI API Key (optional if in .env)", type="password")
    elif llm_provider == "openrouter":
        llm_model = st.selectbox("Model Name", [
            "anthropic/claude-haiku-4.5",
            "anthropic/claude-sonnet-4.6",
            "anthropic/claude-opus-4.5",
            "meta-llama/llama-3.1-70b-instruct",
            "meta-llama/llama-3.1-8b-instruct",
            "google/gemini-2.0-flash-exp",
            "mistralai/mistral-large-2512",
            "qwen/qwen-2.5-coder-32b-instruct",
            "deepseek/deepseek-chat",
            "openai/gpt-4o-mini"
        ])
        api_key = st.text_input("OpenRouter API Key", type="password")
    else: # ollama
        llm_model = st.selectbox(
            "Model Name",
            OLLAMA_MODEL_OPTIONS,
            index=0,
            help="Cloud tags need `ollama pull <model>` and an Ollama account where applicable.",
        )
        api_key = ""  # Not needed
        st.info(
            "Run Ollama locally on port **11434** (default). "
            "Pull a model first, e.g. `ollama pull deepseek-v3.1:671b-cloud`."
        )

    st.markdown("---")
    st.markdown("### 📁 Data Source")
    
    BUILT_IN_DATASETS = {
        "Titanic Passengers 🚢": "titanic.csv",
        "Iris Flowers 🌸": "iris.csv"
    }

    data_source_opt = st.radio("Choose Data Source", ["Upload CSV/Excel", "Use Built-in Dataset"], horizontal=True)
    
    st.markdown("---")
    st.markdown("### 🔧 Processing Pipeline")
    use_enhanced_pipeline = st.checkbox(
        "Use Enhanced Data Processing Pipeline (Recommended)",
        value=True,
        help="Enhanced pipeline provides better data quality scoring, multi-format support, and advanced cleaning"
    )
    
    file_name = None
    file_bytes = None
    
    if data_source_opt == "Upload CSV/Excel":
        uploaded_file = st.file_uploader("Upload CSV/Excel", type=["csv", "xlsx"])
        if uploaded_file is not None:
            file_name = uploaded_file.name
            file_bytes = uploaded_file.getvalue()
    else:
        selected_builtin = st.selectbox("Select a Dataset", list(BUILT_IN_DATASETS.keys()))
        if selected_builtin:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            dataset_path = os.path.join(base_dir, "datasets", BUILT_IN_DATASETS[selected_builtin])
            if os.path.exists(dataset_path):
                file_name = BUILT_IN_DATASETS[selected_builtin]
                with open(dataset_path, "rb") as f:
                    file_bytes = f.read()
            else:
                st.error(f"Cannot find {dataset_path}")

    if file_bytes is not None:
        if st.button("Initialize Workspace", width='stretch', type="primary"):
            with st.spinner("Processing dataset..."):
                files = {"file": (file_name, file_bytes, "text/csv")}
                data = {
                    "use_enhanced": use_enhanced_pipeline,
                    "preprocessing_config": "{}",
                    "cleaning_config": "{}"
                }
                try:
                    # Choose endpoint based on pipeline selection
                    endpoint = "/upload-enhanced" if use_enhanced_pipeline else "/upload"
                    response = requests.post(f"{API_BASE_URL}{endpoint}", files=files, data=data)
                    if response.status_code == 200:
                        data = response.json()
                        st.session_state.dataset_id = data["dataset_id"]
                        st.session_state.dataset_name = data["filename"]
                        st.session_state.is_dirty = data.get("is_dirty", False)
                        st.session_state.original_id = data.get("original_id")
                        st.session_state.cleaned_id = data.get("cleaned_id")
                        st.session_state.cleaning_summary = data.get("cleaning_summary", "")
                        st.session_state.recommended_features = data.get("recommended_features", [])
                        st.session_state.pipeline_used = data.get("pipeline_used", "unknown")
                        st.session_state.data_quality_score = data.get("data_quality_score")
                        
                        pipeline_emoji = "🌟" if use_enhanced_pipeline else "📊"
                        st.success(f"{pipeline_emoji} Successfully loaded {data['filename']} using {data.get('pipeline_used', 'unknown')} pipeline!")
                        
                        # Always surface a professional cleaning report
                        cleaning_summary = data.get("cleaning_summary", "")
                        recommended_features = data.get("recommended_features", [])

                        if cleaning_summary:
                            # Reset streaming flag for the new cleaning report
                            st.session_state["cleaning_report_streamed"] = False

                            msg_lines = [f"🧹 **Data Cleaning Report for `{data['filename']}`**"]
                            msg_lines.append("")
                            msg_lines.append(cleaning_summary)

                            if recommended_features:
                                msg_lines.append("")
                                msg_lines.append("**Recommended Features for Modeling:**")
                                for feat in recommended_features:
                                    msg_lines.append(f"- {feat}")

                            msg_content = "\n".join(msg_lines)

                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": msg_content,
                                "type": "cleaning_report",
                                # Do not duplicate in a separate thought_process field,
                                # since content already shows the full cleaning report.
                                "cleaned_file_id": data.get("cleaned_id")
                            })
                        
                        # Automatically fetch suggested questions
                        try:
                            prompt_payload = {
                                "dataset_id": data["dataset_id"],
                                "query": "", # Empty query triggers suggestion generation
                                "provider": llm_provider,
                                "model_name": llm_model,
                                "api_key": api_key
                            }
                            sugg_res = requests.post(f"{API_BASE_URL}/ask", json=prompt_payload)
                            if sugg_res.status_code == 200:
                                suggestions = sugg_res.json().get("suggested_questions", [])
                                if suggestions:
                                    sugg_text = "**Here are some questions you can ask about this dataset:**\n\n"
                                    for s in suggestions:
                                        sugg_text += f"- {s}\n"
                                    
                                    st.session_state.messages.append({
                                        "role": "assistant",
                                        "content": sugg_text
                                    })
                        except Exception as suggestion_err:
                            pass # Fail silently if suggestions can't be generated
                            
                    else:
                        st.error(f"Error uploading file: {response.text}")
                except Exception as e:
                    st.error(f"Connection error: Make sure the backend is running. {e}")

    st.markdown("---")
    if st.session_state.dataset_name:
        pipeline_info = st.session_state.get("pipeline_used", "unknown")
        pipeline_emoji = "🌟" if "enhanced" in pipeline_info.lower() else "📊"
        st.success(f"🟢 **Active Workspace:** `{st.session_state.dataset_name}` ({pipeline_emoji} {pipeline_info.title()})")
        
        # Display data quality score for enhanced pipeline
        if st.session_state.get("data_quality_score") and "enhanced" in pipeline_info.lower():
            quality_score = st.session_state.data_quality_score
            overall_score = quality_score.get("overall", 0)
            grade = quality_score.get("grade", "N/A")
            
            # Color code the quality score
            if overall_score >= 90:
                color = "🟢"
            elif overall_score >= 80:
                color = "🟡"
            elif overall_score >= 70:
                color = "🟠"
            else:
                color = "🔴"
            
            st.info(f"{color} **Data Quality Score:** {overall_score:.1f}/100 ({grade})")
            
            with st.expander("📊 Detailed Quality Metrics"):
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Completeness", f"{quality_score.get('completeness', 0):.1f}%")
                    st.metric("Uniqueness", f"{quality_score.get('uniqueness', 0):.1f}%")
                with col2:
                    st.metric("Consistency", f"{quality_score.get('consistency', 0):.1f}%")
                    st.metric("Validity", f"{quality_score.get('validity', 0):.1f}%")
        
        if st.session_state.get("is_dirty"):
            st.warning("⚠️ Data contains missing values or duplicates.")
            version_choice = st.radio(
                "Choose Data Version:",
                ["Use Auto-Cleaned Data", "Use Original Dirty Data"],
                index=0
            )
            if version_choice == "Use Auto-Cleaned Data" and st.session_state.cleaned_id:
                st.session_state.dataset_id = st.session_state.cleaned_id
            else:
                st.session_state.dataset_id = st.session_state.original_id

        # Surface recommended modeling features from the cleaning pipeline
        recommended_feats = st.session_state.get("recommended_features") or []
        if recommended_feats:
            with st.expander("📌 Recommended Features for Modeling"):
                st.markdown(
                    "These columns were identified as high-value features "
                    "after cleaning (excluding ID-like and low-information columns):"
                )
                for feat in recommended_feats:
                    st.markdown(f"- `{feat}`")

        # Modeling tools: build and download a model-ready dataset
        with st.expander("🧪 Modeling Tools"):
            st.markdown(
                "Create a model-ready version of the currently selected dataset "
                "(cleaned, ID-like columns removed, categorical features one-hot "
                "encoded, numeric features scaled)."
            )
            if st.button("Create Model-Ready Dataset", width='stretch'):
                if not st.session_state.dataset_id:
                    st.error("Please initialize a dataset first.")
                else:
                    with st.spinner("Building model-ready dataset..."):
                        try:
                            res = requests.get(f"{API_BASE_URL}/model-ready/{st.session_state.dataset_id}")
                            if res.status_code == 200:
                                payload = res.json()
                                st.session_state.model_ready_csv = payload.get("csv", "")
                                st.session_state.model_ready_columns = payload.get("columns", [])
                                st.session_state.model_ready_metadata = payload.get("metadata", {})
                                st.success("Model-ready dataset generated. You can download it below.")
                            else:
                                st.error(f"Failed to create model-ready dataset: {res.text}")
                        except Exception as e:
                            st.error(f"Connection error while creating model-ready dataset: {e}")

            model_ready_csv = st.session_state.get("model_ready_csv", "")
            if model_ready_csv:
                st.download_button(
                    "📥 Download Model-Ready CSV",
                    data=model_ready_csv,
                    file_name=f"model_ready_{st.session_state.dataset_name or 'dataset'}.csv",
                    mime="text/csv",
                    width='stretch',
                )

                cols = st.session_state.get("model_ready_columns") or []
                if cols:
                    st.markdown("**Model features included:**")
                    st.text(", ".join(cols))

                meta = st.session_state.get("model_ready_metadata") or {}
                if meta:
                    st.markdown(
                        f"_Rows: {meta.get('final_row_count', 'N/A')}, "
                        f"Features: {meta.get('final_feature_count', 'N/A')}_"
                    )
    else:
        st.warning("🟡 No dataset loaded. Please upload a file to begin.")

# --- Main App ---
st.markdown('<div class="main-title">AI Data Analyst</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Interact with your data using natural language, powered by dynamic code execution.</div>', unsafe_allow_html=True)

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        # Cleaning report: stream once in the main chat area
        if (
            message.get("type") == "cleaning_report"
            and not st.session_state.get("cleaning_report_streamed", False)
        ):
            placeholder = st.empty()
            streamed = ""
            for ch in message["content"]:
                streamed += ch
                placeholder.markdown(streamed)
                time.sleep(0.005)
            st.session_state["cleaning_report_streamed"] = True
        else:
            st.markdown(message["content"])
        
        # Display cleaning thought process if present
        if "thought_process" in message and message["thought_process"]:
            with st.expander("🧠 View Cleaning Thought Process"):
                st.markdown(message["thought_process"])
                
        # Display download link if present
        if "cleaned_file_id" in message and message["cleaned_file_id"]:
            dl_url = f"{API_BASE_URL}/download/{message['cleaned_file_id']}"
            st.markdown(f"**[📥 Download Cleaned Dataset]({dl_url})**")
            
        # Display the insight text if it's an assistant message
        if "insights" in message and message["insights"]:
            st.info(f"✨ **Key Insight:** \n{message['insights']}")
            
        # Display Plotly figure if it exists
        if "plotly_figure" in message and message["plotly_figure"]:
            try:
                fig = pio.from_json(message["plotly_figure"])
                st.plotly_chart(fig, width='stretch')
            except Exception as e:
                st.error(f"Failed to render interactive chart: {e}")

        # Display static matplotlib/seaborn image fallback
        elif "figure" in message and message["figure"]:
            img_data = message["figure"].split(",")[1]
            st.image(base64.b64decode(img_data), use_column_width=True)
            
        # Display code snippet and dataset info in columns
        row1_col1, row1_col2 = st.columns(2)
        if "code" in message and message["code"]:
            with row1_col1.expander("📝 View Analysis Code"):
                st.code(message["code"], language="python")
                
        if "dataset_info" in message and message["dataset_info"]:
            with row1_col2.expander("📊 View Dataset Metadata"):
                st.text(message["dataset_info"])

# --- Interaction Mode ---
st.markdown("---")
col1, col2 = st.columns([1, 2])
with col1:
    interaction_mode = st.radio(
        "🧠 Choose Interaction Mode:", 
        ["📊 Data Analysis (Code & Charts)", "💬 Chat (Discuss Insights)"], 
        horizontal=True
    )
    chat_mode = "chat" if "Chat" in interaction_mode else "analysis"

# React to user input
prompt_placeholder = "Ask a quantitative query (e.g., 'Plot a heatmap of correlations')..." if chat_mode == "analysis" else "Ask a follow up question about the insights or charts above..."
if prompt := st.chat_input(prompt_placeholder):
    if not st.session_state.dataset_id:
        st.error("Please upload and initialize a dataset from the sidebar first.")
    else:
        # User message
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Assistant response
        with st.chat_message("assistant"):
            with st.spinner("Running AI analysis pipeline..."):
                try:
                    # Build chat history payload (strip out huge images/code, just keep conversational text context)
                    chat_history_payload = []
                    for msg in st.session_state.messages:
                        txt_content = msg.get("content", "")
                        if msg["role"] == "assistant":
                            parts = []
                            if msg.get("insights"):
                                parts.append(f"Insights output: {msg['insights']}")
                            if msg.get("code"):
                                parts.append(f"Python Code used to generate the graph/data:\n```python\n{msg['code']}\n```")
                            if parts:
                                txt_content = "\n\n".join(parts)
                            
                        chat_history_payload.append({"role": msg["role"], "content": txt_content})

                    if chat_mode == "chat":
                        # True streaming for conversational mode
                        payload = {
                            "dataset_id": st.session_state.dataset_id,
                            "query": prompt,
                            "provider": llm_provider,
                            "model_name": llm_model,
                            "api_key": api_key,
                            "chat_history": chat_history_payload,
                        }
                        response = requests.post(
                            f"{API_BASE_URL}/chat-stream",
                            json=payload,
                            stream=True,
                        )
                        if response.status_code != 200:
                            err_msg = f"API Error (stream): {response.text}"
                            st.error(err_msg)
                            st.session_state.messages.append({"role": "assistant", "content": err_msg})
                        else:
                            placeholder = st.empty()
                            full_text = ""
                            for chunk in response.iter_content(chunk_size=1024):
                                if not chunk:
                                    continue
                                text = chunk.decode("utf-8", errors="ignore")
                                full_text += text
                                placeholder.markdown(full_text)

                            # Save streamed response to history (treat as plain chat, no separate 'insights')
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": full_text,
                                "code": "",
                                "dataset_info": "",
                                "figure": None,
                                "plotly_figure": None
                            })
                    else:
                        # Analysis mode (code + charts), non-streaming but with insight typing effect
                        payload = {
                            "dataset_id": st.session_state.dataset_id,
                            "query": prompt,
                            "provider": llm_provider,
                            "model_name": llm_model,
                            "api_key": api_key,
                            "chat_history": chat_history_payload,
                            "chat_mode": chat_mode
                        }
                        start_time = time.time()
                        res = requests.post(f"{API_BASE_URL}/ask", json=payload)
                        end_time = time.time()
                        
                        if res.status_code == 200:
                            data = res.json()
                            
                            # Only show "Analysis Completed" if it actually did data execution (i.e., not a purely conversational chat)
                            if data.get("code") or data.get("figure") or data.get("plotly_figure"):
                                st.markdown(f"**Analysis Completed** `({round(end_time - start_time, 2)}s)`")
                            
                            insights = data.get("insights", "")
                            if insights:
                                # Streaming-style rendering of insight text for better UX
                                insight_placeholder = st.empty()
                                streamed = ""
                                for ch in insights:
                                    streamed += ch
                                    insight_placeholder.info(f"✨ **Key Insight:** \n{streamed}")
                                    time.sleep(0.01)
                                
                            # Try parsing plotly figure first
                            plotly_fig = data.get("plotly_figure")
                            figure = data.get("figure")
                            
                            if plotly_fig:
                                try:
                                    fig = pio.from_json(plotly_fig)
                                    st.plotly_chart(fig, width='stretch')   
                                except Exception as e:
                                    st.error(f"Failed to render interactive chart: {e}")
                            # Fallback to static matplotlib
                            elif figure:
                                img_data = figure.split(",")[1]
                                st.image(base64.b64decode(img_data), use_column_width=True)

                            gen_code = data.get("code", "")
                            dataset_info = data.get("dataset_info_snippet", "")
                            
                            c1, c2 = st.columns(2)
                            if gen_code:
                                with c1.expander("📝 View Analysis Code"):
                                    st.code(gen_code, language="python")
                            if dataset_info:
                                with c2.expander("📊 View Dataset Metadata"):
                                    st.text(dataset_info)
                                
                            # Save to history
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": f"**Analysis Completed** `({round(end_time - start_time, 2)}s)`",
                                "insights": insights,
                                "code": gen_code,
                                "dataset_info": dataset_info,
                                "figure": figure,
                                "plotly_figure": plotly_fig
                            })
                        else:
                            err_msg = f"API Error: {res.text}"
                            st.error(err_msg)
                            st.session_state.messages.append({"role": "assistant", "content": err_msg})

                except Exception as e:
                    err_msg = f"Connection Interrupted: {e}"
                    st.error(err_msg)
                    st.session_state.messages.append({"role": "assistant", "content": err_msg})
