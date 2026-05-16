FROM python:3.10-slim

RUN pip install fastapi uvicorn pydantic pandas numpy --no-cache-dir

WORKDIR /app
COPY sandbox_server.py .

# 安全加固：创建非 root 用户
RUN useradd -m sandboxuser
USER sandboxuser

EXPOSE 8000
CMD ["python", "sandbox_server.py"]