FROM python:3.12-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.5.29 /uv /uvx /bin/

# Install build dependencies and uv
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Sync the project into a new environment, using the frozen lockfile
WORKDIR /app

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen

COPY . .

EXPOSE 8000

CMD ["uv", "run", "streamlit", "run", "main.py", "--server.port", "8000"]
