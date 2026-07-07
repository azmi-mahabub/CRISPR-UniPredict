FROM pytorch/pytorch:2.0.0-cuda11.8-cudnn8-runtime

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy web app requirements
COPY web_app/requirements.txt web_app/requirements.txt
RUN pip install --no-cache-dir -r web_app/requirements.txt

# Copy application code
COPY models/ models/
COPY api/ api/
COPY web_app/ web_app/
COPY configs/ configs/
COPY utils/ utils/
COPY training/ training/
COPY scripts/ scripts/

# Create necessary directories
RUN mkdir -p /app/models/checkpoints \
    && mkdir -p /app/results \
    && mkdir -p /app/logs

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose ports
EXPOSE 8000 8501

# Create startup script
RUN echo '#!/bin/bash\n\
echo "Starting CRISPR-UniPredict services..."\n\
echo "API will be available at http://localhost:8000"\n\
echo "Web UI will be available at http://localhost:8501"\n\
\n\
# Start API in background\n\
uvicorn api.inference_api:app --host 0.0.0.0 --port 8000 &\n\
API_PID=$!\n\
\n\
# Wait for API to start\n\
sleep 5\n\
\n\
# Start Streamlit\n\
streamlit run web_app/app.py --server.port=8501 --server.address=0.0.0.0\n\
' > /app/start.sh && chmod +x /app/start.sh

CMD ["/app/start.sh"]
