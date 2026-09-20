FROM python:3.10-slim

# Install system packages (ffmpeg is required for Whisper & Pydub)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt uvicorn fastapi python-multipart

# Copy project code
COPY . .

# Create necessary runtime directories
RUN mkdir -p data downloads static

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Expose FastAPI (8000) and Streamlit (8501)
EXPOSE 8000
EXPOSE 8501

# Default startup command (FastAPI server hosting Web UI) - dynamic port for PaaS support
CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port ${PORT:-8000}"]

