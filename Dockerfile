FROM python:3.11-slim

WORKDIR /app

# Install gcc for any C extensions (psutil, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY revia_core_py/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt psutil

# Copy core server source
COPY revia_core_py/ .

# Persistent data and log directories
RUN mkdir -p data logs

EXPOSE 8123 8124

CMD ["python", "core_server.py"]
