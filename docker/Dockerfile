# SmartRAG Docker Image - Production Ready
# Multi-stage build for optimized image size and security

# ============================================
# Stage 1: Builder - Install and cache models
# ============================================
FROM python:3.10-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Pre-download Hugging Face models to cache directory
RUN mkdir -p /build/.cache/huggingface
ENV HF_HOME=/build/.cache/huggingface
RUN python3 -c "from transformers import BlipProcessor, BlipForConditionalGeneration; \
    processor = BlipProcessor.from_pretrained('Salesforce/blip-image-captioning-base'); \
    model = BlipForConditionalGeneration.from_pretrained('Salesforce/blip-image-captioning-base'); \
    print('✓ BLIP model cached successfully')"

# Pre-download Whisper model
RUN python3 -c "import whisper; \
    model = whisper.load_model('base'); \
    print('✓ Whisper model cached successfully')"

# Pre-download CLIP model for image embeddings
RUN python3 -c "from transformers import CLIPProcessor, CLIPModel; \
    processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32'); \
    model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32'); \
    print('✓ CLIP model cached successfully')"

# ============================================
# Stage 2: Runtime - Final production image
# ============================================
FROM python:3.10-slim

# Add metadata labels
LABEL maintainer="SmartRAG Team"
LABEL version="1.0"
LABEL description="Multimodal RAG System with Ollama"

# Install runtime system dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    curl \
    zstd \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Create non-root user for security
RUN groupadd -r smartrag --gid=1000 && \
    useradd -r -g smartrag --uid=1000 --create-home --shell /bin/bash smartrag

# Set working directory
WORKDIR /app

# Copy Python packages from builder
COPY --from=builder --chown=smartrag:smartrag /root/.local /home/smartrag/.local

# Copy cached models from builder
COPY --from=builder --chown=smartrag:smartrag /build/.cache /home/smartrag/.cache

# Update PATH to use user-installed packages
ENV PATH=/home/smartrag/.local/bin:$PATH
ENV HF_HOME=/home/smartrag/.cache/huggingface
ENV TRANSFORMERS_CACHE=/home/smartrag/.cache/huggingface

# Copy application code
COPY --chown=smartrag:smartrag . .

# Create necessary directories with proper permissions
RUN mkdir -p vector_db temp_uploads user_data logs data && \
    chown -R smartrag:smartrag vector_db temp_uploads user_data logs data

# Removed Streamlit port usage

# Expose ports
EXPOSE 8501 11434

# Expose FastAPI port
EXPOSE 8000

# Create startup script
RUN echo '#!/bin/bash\n\
    set -euo pipefail\n\
    \n\
    # Trap signals for graceful shutdown\n\
    trap "echo Shutting down...; kill -TERM $OLLAMA_PID 2>/dev/null; exit 0" SIGTERM SIGINT\n\
    \n\
    echo "========================================"\n\
    echo "SmartRAG Production Startup"\n\
    echo "========================================"\n\
    \n\
    # Start Ollama service\n\
    echo "[1/5] Starting Ollama service..."\n\
    OLLAMA_HOST=0.0.0.0:11434 ollama serve &\n\
    OLLAMA_PID=$!\n\
    \n\
    # Wait for Ollama to be ready with timeout\n\
    echo "[2/5] Waiting for Ollama to be ready..."\n\
    MAX_RETRIES=30\n\
    RETRY_COUNT=0\n\
    while ! curl -s http://localhost:11434/api/tags >/dev/null 2>&1; do\n\
    RETRY_COUNT=$((RETRY_COUNT + 1))\n\
    if [ $RETRY_COUNT -ge $MAX_RETRIES ]; then\n\
    echo "ERROR: Ollama failed to start after ${MAX_RETRIES} seconds"\n\
    exit 1\n\
    fi\n\
    echo "  Waiting for Ollama... (${RETRY_COUNT}/${MAX_RETRIES})"\n\
    sleep 1\n\
    done\n\
    echo "  ✓ Ollama is ready"\n\
    \n\
    # Pull required models with retry logic\n\
    echo "[3/5] Pulling required Ollama models..."\n\
    pull_model() {\n\
    local model=$1\n\
    local max_attempts=3\n\
    local attempt=1\n\
    \n\
    while [ $attempt -le $max_attempts ]; do\n\
    echo "  Pulling $model (attempt $attempt/$max_attempts)..."\n\
    if ollama pull $model; then\n\
    echo "  ✓ $model downloaded successfully"\n\
    return 0\n\
    fi\n\
    attempt=$((attempt + 1))\n\
    [ $attempt -le $max_attempts ] && sleep 5\n\
    done\n\
    \n\
    echo "  ⚠ Warning: Failed to pull $model after $max_attempts attempts"\n\
    return 1\n\
    }\n\
    \n\
    pull_model "llama3.1:8b"\n\
    pull_model "nomic-embed-text"\n\
    \n\
    echo "Starting FastAPI server..."\n\
    exec uvicorn api:app --host 0.0.0.0 --port 8000\n\
    ' > /app/start.sh && chmod +x /app/start.sh

# Set environment variables
ENV OLLAMA_HOST=http://localhost:11434

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Run startup script
CMD ["/bin/bash", "/app/start.sh"]
