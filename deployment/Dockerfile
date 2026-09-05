# Base image: slim Python 3.11. "slim" = minimal OS packages, keeps image
# size down versus the full python:3.11 image (~120MB vs ~1GB base).
FROM python:3.11-slim

WORKDIR /app

# System-level libraries opencv-python-headless needs to actually run
# (not just import) inside a minimal Linux image - these aren't installed
# by pip, they're OS packages.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install CPU-only torch explicitly BEFORE the general requirements install.
# Without this, pip may resolve a much larger CUDA-enabled torch build -
# wasted size and download time, since our design doc targets CPU inference
# (no GPU assumed at deployment, matching the latency test we just validated).
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Copy only what the running service actually needs - not the whole repo
# (not data/, notebooks/, tests/). Smaller image, faster builds, and no
# risk of accidentally shipping training data in a production image.
COPY src/ src/
COPY models/ models/

EXPOSE 8000

# 0.0.0.0 (not 127.0.0.1/localhost) is required here - inside a container,
# binding to localhost would make the server unreachable from OUTSIDE the
# container, since "localhost" would only mean "inside this container."
CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]