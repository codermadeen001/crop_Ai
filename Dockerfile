# Dockerfile
FROM python:3.11-slim  

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt --no-cache-dir

COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED 1
ENV PORT 8000

# Run Gunicorn with reduced workers (memory optimization)
CMD ["gunicorn", "--workers=2", "--threads=4", "--bind", "0.0.0.0:8000", "--timeout", "120", "backend.wsgi:application"]