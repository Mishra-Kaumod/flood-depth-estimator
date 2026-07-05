FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir .

# Copy only production modules
COPY app.py tasks.py temporal_aggregator.py mc_dropout.py schemas.py ./
COPY src/ src/
COPY templates/ templates/
COPY static/ static/
COPY config/ config/

# Model weights (supplied at runtime via volume mount or ENV)
ENV MODEL_PATH=/models/flood_model_v6.1.pth
ENV FLASK_ENV=production

EXPOSE 5000
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "60", "app:app"]
