# SmartRAG Docker Image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    git \
    curl \
    zstd \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download Hugging Face models to avoid runtime downloads
RUN python3 -c "from transformers import BlipProcessor, BlipForConditionalGeneration; \
    processor = BlipProcessor.from_pretrained('Salesforce/blip-image-captioning-base'); \
    model = BlipForConditionalGeneration.from_pretrained('Salesforce/blip-image-captioning-base'); \
    print('BLIP model cached successfully')"

# Pre-download Whisper model
RUN python3 -c "import whisper; \
    model = whisper.load_model('base'); \
    print('Whisper model cached successfully')"

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p vector_db temp_uploads user_data logs data

# Removed Streamlit port usage

# Expose Ollama port
EXPOSE 11434

# Expose FastAPI port
EXPOSE 8000

# Create startup script
RUN echo '#!/bin/bash\n\
    set -e\n\
    \n\
    echo "Starting Ollama service..."\n\
    ollama serve &\n\
    OLLAMA_PID=$!\n\
    \n\
    echo "Waiting for Ollama to start..."\n\
    sleep 5\n\
    \n\
    echo "Pulling Llama 3.1 8B model..."\n\
    ollama pull llama3.1:8b || echo "Model already exists or pull failed"\n\
    \n\
    echo "Pulling Nomic Embed Text model..."\n\
    ollama pull nomic-embed-text || echo "Model already exists or pull failed"\n\
    \n\
    echo "Verifying Hugging Face models..."\n\
    python3 -c "from transformers import BlipProcessor; BlipProcessor.from_pretrained('"'"'Salesforce/blip-image-captioning-base'"'"')" || echo "BLIP model check failed"\n\
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
CMD ["/app/start.sh"]
