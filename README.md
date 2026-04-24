# AI Data Analyst Agent

A production-ready AI agent system that allows users to upload datasets and ask natural language questions. The system analyzes data, generates Python code, executes it safely, and returns results with visualizations and business insights.

## 🚀 Quick Start

### Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys
```

### Run the Application
```bash
# Start the server
python main.py

# Or alternatively
python -m uvicorn backend.main:app --reload --port 8000
```

### Access
- **Backend API**: http://localhost:8000
- **Frontend**: http://localhost:8501 (after starting Streamlit)

## 📁 Project Structure

```
ai-data-analyst-agent/
├── main.py                    # 🚀 Application launcher
├── requirements.txt           # 📋 Dependencies
├── .env.example              # 🔑 Environment template
│
├── docs/                     # 📚 Documentation
│   ├── README.md             # This file
│   └── ENHANCED_PIPELINES.md # Enhanced pipelines guide
│
├── tests/                    # 🧪 Test suite
│   ├── README.md             # Test documentation
│   └── test_enhanced_pipelines.py
│
├── backend/                  # 📦 Backend module
│   ├── main.py               # FastAPI application
│   └── pipelines/            # 🔄 Data processing
│       ├── core/             # 🌟 Enhanced pipelines (recommended)
│       └── legacy/           # 📊 Original pipelines (backward compat)
│
├── frontend/                 # 🎨 Streamlit interface
└── datasets/                 # 📁 Uploaded datasets
```

## 🔧 Features

### Enhanced Data Processing (Recommended)
- **Multi-format support**: US/European numbers, currencies, scientific notation
- **Advanced missing value analysis**: Pattern detection and correlation analysis
- **Multi-method outlier detection**: IQR, Z-score, Isolation Forest
- **Data quality scoring**: Comprehensive A-F grading system
- **Intelligent encoding**: Automatic strategy selection based on data characteristics

### AI Agent Capabilities
- **Natural language understanding**: Process user queries about data
- **Code generation**: Write pandas/matplotlib/plotly code automatically
- **Safe execution**: Run generated code in isolated environment
- **Visualization**: Create charts and graphs based on queries
- **Business insights**: Generate actionable insights from analysis

## 📡 API Endpoints

### Enhanced Pipeline (Recommended)
```bash
POST /upload-enhanced
- Advanced cleaning and preprocessing
- Data quality scoring
- Configurable parameters
- Comprehensive reporting
```

### Standard Endpoints
```bash
GET  /                    # Health check
POST /upload             # Basic data upload
POST /ask                # Query analysis
POST /chat-stream        # Streaming chat
GET  /download/{id}      # Download dataset
GET  /model-ready/{id}   # Get model-ready data
```

## 🧪 Testing

```bash
# Run tests
python tests/test_enhanced_pipelines.py

# Or with pytest
python -m pytest tests/ -v

# With coverage
python -m pytest tests/ --cov=backend
```

## 💻 Usage Examples

### Enhanced Data Processing
```python
from backend.pipelines.core import preprocess_dataset_enhanced, clean_dataset_enhanced

# Clean data
cleaned = clean_dataset_enhanced(df)
print(f"Data quality score: {cleaned['data_quality_score']['overall']:.1f}/100")

# Preprocess for modeling
model_ready = preprocess_dataset_enhanced(cleaned['cleaned_df'])
features = model_ready['recommended_features']
```

### API Usage
```python
import requests

# Upload dataset with enhanced processing
files = {'file': open('data.csv', 'rb')}
response = requests.post('http://localhost:8000/upload-enhanced', files=files)

# Ask questions about the data
data = response.json()
query_data = {
    'dataset_id': data['dataset_id'],
    'query': 'Show me the correlation between all numeric columns',
    'provider': 'openai',
    'model_name': 'gpt-4o-mini'
}
response = requests.post('http://localhost:8000/ask', json=query_data)
```

## 🌟 Enhanced vs Original Pipelines

| Feature | Original | Enhanced |
|---------|----------|----------|
| Data Type Detection | Basic | Advanced (15+ formats) |
| Missing Value Analysis | Simple | Pattern analysis |
| Outlier Detection | IQR only | 3 methods + consensus |
| Quality Scoring | ❌ | ✅ A-F grades |
| International Support | ❌ | ✅ European/US formats |
| Configuration | ❌ | ✅ Fully configurable |

## 🛠️ Configuration

### Enhanced Pipeline Configuration
```python
config = {
    "missing_threshold": 0.3,
    "outlier_method": "zscore",
    "scaling_method": "robust",
    "feature_selection": True,
    "data_quality_score": True
}

result = preprocess_dataset_enhanced(df, config)
```

### Environment Variables
```bash
OPENAI_API_KEY=your_openai_api_key
OPENROUTER_API_KEY=your_openrouter_api_key
```

## 🚀 Deployment

### Docker
```bash
docker build -t ai-data-analyst .
docker run -p 8000:8000 -p 8501:8501 --env-file .env ai-data-analyst
```

### Production
- Use enhanced pipelines for better data quality
- Configure appropriate resource limits
- Set up monitoring and logging
- Use HTTPS in production

## 🤖 Tech Stack

- **Backend**: Python, FastAPI, LangGraph, LangChain
- **Data Processing**: Pandas, NumPy, Scikit-learn, SciPy
- **AI**: OpenAI GPT, OpenRouter (optional)
- **Frontend**: Streamlit
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Infrastructure**: Docker, Uvicorn

## 📚 Documentation

- **[Enhanced Pipelines Guide](docs/ENHANCED_PIPELINES.md)** - Detailed technical documentation
- **[Test Suite](tests/README.md)** - Testing documentation and examples

---

**🎯 For production use, the enhanced pipelines in `backend/pipelines/core/` are recommended for superior data quality and processing capabilities.**
