FROM python:3.11-slim

# System deps for faiss-cpu / sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ libgomp1 git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer cache)
COPY code/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the embedding model so the image is self-contained
RUN python -c "from sentence_transformers import SentenceTransformer; \
               SentenceTransformer('all-MiniLM-L6-v2')"

# Copy the full repo
COPY . /app

WORKDIR /app/code

# Default: process the tickets and write output
ENTRYPOINT ["python", "main.py"]
CMD ["--tickets", "../support_tickets/support_tickets.csv"]
