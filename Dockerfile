# ── Credit Approval Environment — Docker ──
# This container runs the FastAPI environment server ONLY.
# inference.py runs OUTSIDE this container (in the evaluator).
# Keep this image lean: no openai, no openenv-core, no gradio.
FROM python:3.12-slim

WORKDIR /app

# Install ONLY server dependencies (fastapi, uvicorn, pydantic)
COPY server/requirements.txt server/requirements.txt
RUN pip install --no-cache-dir -r server/requirements.txt

# Copy environment code
COPY models.py .
COPY server/ server/
COPY __init__.py .
COPY openenv.yaml .
COPY pyproject.toml .

# OpenEnv SDK default port is 7860 for HF Spaces
# The SDK reads the port from openenv.yaml app.port
EXPOSE 7860

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
