FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Fetch the full-size embedding/reranker models at BUILD time (Cloud Build has real resources,
# unlike Render's constrained free-tier build machine) so they're baked into the image and load
# with local_files_only=True at request time.
RUN python -m app.download_models

# Build the FAISS/BM25 index from whatever PDFs are committed under data/raw/ — also at build
# time, so the deployed image is self-contained. Cloud Run has no persistent disk between
# revisions, same reasoning as the Render build: bake it in rather than rely on runtime storage.
RUN python -m app.ingest

# Cloud Run injects $PORT (default 8080) — shell form so it's expanded at container start, not
# treated literally. Single worker: model state is per-process: multiple workers would each load
# their own full copy of both models, which is wasteful even with Cloud Run's generous memory.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1"]
