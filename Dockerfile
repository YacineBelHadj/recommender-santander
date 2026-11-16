FROM python:3.12-slim
WORKDIR /app
ENV PORT=8000 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 POLARS_MAX_THREADS=2
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir --no-deps .
COPY predict.py ./
COPY artifacts/model.joblib ./artifacts/model.joblib
EXPOSE 8000
CMD ["sh", "-c", "exec uvicorn predict:app --host 0.0.0.0 --port ${PORT:-8000}"]
