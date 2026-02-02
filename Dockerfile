FROM python:3.11-slim

# CHANGE: Set the working directory directly to where the code is
WORKDIR /app/Backend

# Copy requirements (adjusting path because we are now inside /app/Backend)
COPY Backend/requirements.txt .

RUN apt-get update && \
    apt-get install -y gcc g++ unixodbc-dev libpq-dev curl && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# CHANGE: Copy the code into the current directory (.)
COPY Backend/ .

EXPOSE 8000

# CHANGE: Simplified CMD. No "Backend." prefix needed anymore.
# Because we are already inside the Backend folder, imports just work.
CMD ["uvicorn", "fastapi_app:app", "--host", "0.0.0.0", "--port", "8000"]