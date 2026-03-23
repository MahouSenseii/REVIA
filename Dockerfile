FROM python:3.11-slim

WORKDIR /app

# Install gcc for any C extensions (psutil, etc.) and curl for health checks.
RUN apt-get update && apt-get install -y --no-install-recommends gcc curl && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY revia_core_py/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt psutil

# Copy core server source
COPY revia_core_py/ .

# Persistent data and log directories
RUN mkdir -p data logs

# Run as a non-root user to reduce the attack surface.
RUN adduser --disabled-password --gecos "" revia && \
    chown -R revia:revia /app
USER revia

EXPOSE 8123 8124

# Confirm the REST endpoint is reachable.
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -sf http://localhost:8123/api/status || exit 1

CMD ["python", "core_server.py"]
