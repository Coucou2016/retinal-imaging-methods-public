# CPU-friendly smoke image for the FLAT snapshot (no CUDA, no patient data).
# Everything is copied flat into /app; there are no sub-packages.
FROM python:3.12-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements-flat.txt

# Flat layout: discovery runs from the root itself.
CMD ["python", "-m", "unittest", "discover", "-s", ".", "-p", "test_*.py", "-v"]
