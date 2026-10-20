FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Use HTTPS for Debian package repositories
RUN sed -i 's|http://deb.debian.org|https://deb.debian.org|g' \
    /etc/apt/sources.list.d/debian.sources \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install PyTorch from the locally downloaded wheel
COPY vendor/torch-2.14.1+cpu-cp312-cp312-manylinux_2_28_x86_64.whl \
    /tmp/torch-2.14.1+cpu-cp312-cp312-manylinux_2_28_x86_64.whl

RUN pip install --no-cache-dir --retries 10 --timeout 600 \
    /tmp/torch-2.14.1+cpu-cp312-cp312-manylinux_2_28_x86_64.whl

# Install the remaining dependencies from PyPI
RUN grep -v '^torch==' requirements.txt > /tmp/requirements-api.txt \
    && pip install --no-cache-dir --retries 10 --timeout 600 \
    --index-url https://pypi.org/simple \
    -r /tmp/requirements-api.txt

# Copy application code and trained models
COPY src/ ./src/
COPY models/ ./models/

EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
