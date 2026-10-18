# Build from repository root: docker build -f multiple-search/legal_sandbox/Dockerfile -t legal-sandbox:v2 .
FROM python:3.13-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY agent_tools/sandbox/runner.py /app/runner.py
USER 65534:65534
CMD ["python", "-c", "import time; time.sleep(86400)"]
