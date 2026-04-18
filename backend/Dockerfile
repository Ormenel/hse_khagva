FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for numerical libraries
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ libgomp1 curl && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY backend/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy backend package
COPY backend/ ./backend/

# Copy models package (needed for ML prepayment inference)
COPY models/ ./models/

EXPOSE 8000

CMD ["uvicorn", "backend.api:app", "--host", "0.0.0.0", "--port", "8000"]
