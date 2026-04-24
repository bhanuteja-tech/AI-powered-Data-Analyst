FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose ports for FastAPI (8000) and Streamlit (8501)
EXPOSE 8000 8501

# Start script
# We'll use a simple shell script to run both services
RUN echo "#!/bin/sh\n\
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 &\n\
streamlit run frontend/streamlit_app.py --server.port 8501 --server.address 0.0.0.0\n\
" > start.sh && chmod +x start.sh

CMD ["./start.sh"]
