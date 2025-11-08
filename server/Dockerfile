# ---- Stage 1: Build environment with PyTorch ----
FROM pytorch/pytorch:latest AS base

WORKDIR /app

# Copy app code and models
COPY ./Model ./app/Model
COPY ./app ./app
COPY ./app/requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# ---- Stage 2: Slim runtime ----
FROM python:3.13-slim

WORKDIR /app

# Copy installed environment (torch + deps) from the base stage
COPY --from=base /opt/conda /opt/conda
COPY --from=base /app /app

# Ensure conda binaries are on PATH
ENV PATH="/opt/conda/bin:$PATH"

EXPOSE 8000

# Start the FastAPI app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

