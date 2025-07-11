FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc python3-dev && \
    rm -rf /var/lib/apt/lists/*

# Copy model file FIRST (for caching)
COPY cropAi/best_model.h5 /app/cropAi/best_model.h5

# Verify model exists
RUN ls -la /app/cropAi/best_model.h5 && \
    chmod 644 /app/cropAi/best_model.h5

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt --no-cache-dir

# Copy the rest of the code
COPY . .

# Runtime config
ENV PYTHONUNBUFFERED=1 \
    PORT=8000 \
    TF_CPP_MIN_LOG_LEVEL=2 
# Fixed CMD - either use direct port or shell form for variable expansion
#CMD ["gunicorn", "--workers=2", "--threads=4", "--bind", "0.0.0.0:8000", "--timeout", "120", "backend.wsgi:application"]


# Replace the CMD line with this:
CMD ["gunicorn", "--workers=2", "--bind", "0.0.0.0:$PORT", "backend.wsgi:application"]