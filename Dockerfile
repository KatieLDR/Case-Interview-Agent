# ── Base image ────────────────────────────────────────────────────────────
FROM python:3.11-slim

# ── Environment ────────────────────────────────────────────────────────────
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_VERSION=1.8.2 \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_CREATE=false

ENV PATH="$POETRY_HOME/bin:$PATH"

# ── Install Poetry ─────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y curl && \
    curl -sSL https://install.python-poetry.org | python3 - && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# ── Working directory ──────────────────────────────────────────────────────
WORKDIR /app

# ── Install dependencies ───────────────────────────────────────────────────
COPY pyproject.toml poetry.lock ./
RUN poetry install --no-root --no-interaction --no-ansi

# ── Copy app code ──────────────────────────────────────────────────────────
COPY backend/ ./backend/
COPY frontend/ ./frontend/
COPY .chainlit/ ./.chainlit/
COPY chainlit.md ./

# ── Port ───────────────────────────────────────────────────────────────────
# Cloud Run injects $PORT at runtime — Chainlit must listen on it
ENV PORT=8080 \
    PYTHONPATH=/app
EXPOSE 8080

# ── Start command ──────────────────────────────────────────────────────────
CMD ["sh", "-c", "poetry run chainlit run frontend/app.py --host 0.0.0.0 --port $PORT"]
