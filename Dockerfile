# ── Credit Approval Environment — HF Spaces Docker ──
FROM python:3.12-slim

WORKDIR /app

# Install server dependencies
COPY server/requirements.txt server/requirements.txt
RUN pip install --no-cache-dir -r server/requirements.txt

# Install inference dependencies (openai, openenv-core)
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy environment code
COPY models.py .
COPY server/ server/
COPY __init__.py .
COPY openenv.yaml .
COPY pyproject.toml .
COPY inference.py .
COPY client.py .

# HF Spaces requires port 7860
EXPOSE 7860

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
