FROM python:3.11-slim

LABEL maintainer="Scaler OpenEnv"
LABEL description="SQL Data Quality Environment — OpenEnv-compatible"
LABEL org.opencontainers.image.title="SQL Data Quality Environment"

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency file first for layer caching
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY models.py tasks.py environment.py ./
COPY server/ ./server/
COPY openenv.yaml ./

# Create a non-root user for security
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

EXPOSE 7860

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
