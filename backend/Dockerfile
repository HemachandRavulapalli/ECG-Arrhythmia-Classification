# Dockerfile for Railway deployment

FROM python:3.10-slim


WORKDIR /app

# Install system dependencies for OpenCV, PDF/image processing, and OCR
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 tesseract-ocr tesseract-ocr-eng


# Copy requirements files
COPY requirements.txt .
COPY requirements.deploy.txt .

# Install dependencies needed for serving predictions
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install -r requirements.txt

# Copy application code
COPY . .

# Expose port (Railway sets PORT env var)
EXPOSE 8000

# Start command
CMD ["python3", "app.py"]

