FROM Python:3.11.9-slim

WORKDIR /app

# Install dependencies first to leverage Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the API script and serialized model artifact
COPY app.py model.pkl ./

# Expose the API port
EXPOSE 8000

# Start the ASGI server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]