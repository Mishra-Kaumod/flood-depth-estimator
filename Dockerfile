FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml .
COPY app.py tasks.py mc_dropout.py serve.py ./
COPY src/ src/
COPY config/ config/
COPY models/ models/

RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir .

ENV PYTHONUNBUFFERED=1
ENV FLASK_ENV=production

EXPOSE 5000
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-5000} --workers 2 --timeout 60 app:app"]
