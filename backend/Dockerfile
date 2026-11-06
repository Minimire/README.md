# ─────────────────────────────────────────────────────────────
# Base image — pinned for reproducibility
# ─────────────────────────────────────────────────────────────
FROM python:3.11.9-slim

# Unbuffered logs (so you actually see output in HF/Docker logs),
# no .pyc files, no pip cache.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# ─────────────────────────────────────────────────────────────
# 1. Dependencies FIRST — this layer is cached and only rebuilds
#    when requirements.txt changes.
# ─────────────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ─────────────────────────────────────────────────────────────
# 2. App code AFTER deps — code edits don't invalidate the pip layer.
#    --chown so appuser can write to /app at runtime.
# ─────────────────────────────────────────────────────────────
COPY --chown=appuser:appuser . .

# ─────────────────────────────────────────────────────────────
# 3. Non-root user — created AFTER install & copy, so pip still
#    runs as root, but the app runs unprivileged.
# ─────────────────────────────────────────────────────────────
RUN useradd --create-home appuser
USER appuser

EXPOSE 7860

# ─────────────────────────────────────────────────────────────
# 4. Healthcheck — TCP connect to the port (no route dependency).
#    OPTIONAL: delete this block if you don't want it.
# ─────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import socket,sys; s=socket.socket(); s.settimeout(3); sys.exit(0 if s.connect_ex(('127.0.0.1',7860))==0 else 1)"]

# ─────────────────────────────────────────────────────────────
# 5. Start the app
# ─────────────────────────────────────────────────────────────
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", \
     "--proxy-headers", "--forwarded-allow-ips", "*"]
