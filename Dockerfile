# CPU-friendly smoke image for unit tests / demo training (no CUDA, no patient data).
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt pyproject.toml README.md ./
COPY reti_pioneer ./reti_pioneer
COPY model ./model
COPY dataset ./dataset
COPY utils ./utils
COPY scripts ./scripts
COPY tests ./tests
COPY configs ./configs

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

CMD ["python", "-m", "unittest", "discover", "-s", "tests", "-v"]
