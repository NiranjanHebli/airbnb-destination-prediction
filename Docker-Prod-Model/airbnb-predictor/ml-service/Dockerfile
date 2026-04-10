FROM python:3.11-slim

WORKDIR /app

# Install curl for docker healthcheck, then install python dependencies
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application and model artefacts
COPY main.py .
COPY production_model.pkl .
COPY preprocessor.pkl .
COPY label_encoder.pkl .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
