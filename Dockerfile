FROM python:3.11-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y \
    gcc python3-dev && \
    rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt --no-cache-dir

# Copy model file FIRST (for better caching)
COPY backend/cropAi/best_model.h5 /app/backend/cropAi/best_model.h5

# Verify model exists
RUN ls -la /app/backend/cropAi/best_model.h5 && \
    chmod 644 /app/backend/cropAi/best_model.h5

# Copy the rest of the code
COPY . .

# Runtime config
ENV PYTHONUNBUFFERED=1 \
    PORT=8000 \
    TF_CPP_MIN_LOG_LEVEL=2  

CMD ["gunicorn", "--workers=2", "--threads=4", "--bind", "0.0.0.0:$PORT", "--timeout", "120", "backend.wsgi:application"]