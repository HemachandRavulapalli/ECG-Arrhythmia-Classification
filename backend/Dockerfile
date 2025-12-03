# Dockerfile for Railway deployment
FROM python:3.10-slim

WORKDIR /app

# Copy lightweight deployment requirements for faster installs
COPY requirements.deploy.txt ./requirements.deploy.txt

# Install dependencies needed for serving predictions
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install -r requirements.deploy.txt

# Copy application code
COPY . .

# Expose port (Railway sets PORT env var)
EXPOSE 8000

# Start command
CMD ["python3", "app.py"]

