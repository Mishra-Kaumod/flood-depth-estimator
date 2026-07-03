"""
Django settings for flood_project.
Enterprise-friendly defaults with environment overrides.
"""

from pathlib import Path
import os
import dj_database_url
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

default_runtime_root = Path(r"E:\flood_runtime") if Path("E:\\").exists() else (BASE_DIR / "runtime")
RUNTIME_ROOT = Path(os.getenv("FLOOD_RUNTIME_ROOT", str(default_runtime_root)))
RUNTIME_TMP_DIR = RUNTIME_ROOT / "tmp_uploads"
RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
RUNTIME_TMP_DIR.mkdir(parents=True, exist_ok=True)

SECRET_KEY = os.getenv(
    "DJANGO_SECRET_KEY",
    "django-insecure-(*t=m3jyo(btl(6y8v&1be#os*s+$_od3fw0-i3n^o*v47ngp6",
)
DEBUG = os.getenv("DJANGO_DEBUG", "true").lower() == "true"
ALLOWED_HOSTS = [h.strip() for h in os.getenv("DJANGO_ALLOWED_HOSTS", "*").split(",") if h.strip()]

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "rest_framework",
    "flood_api",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "flood_project.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "flood_project.wsgi.application"

DATABASES = {
    "default": dj_database_url.parse(
        os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR / 'db.sqlite3'}"),
        conn_max_age=600,
    )
}

AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

LANGUAGE_CODE = "en-us"
TIME_ZONE = os.getenv("DJANGO_TIME_ZONE", "UTC")
USE_I18N = True
USE_TZ = True

STATIC_URL = "static/"
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# Async queue configuration (Redis + Celery).
CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")
CELERY_ACCEPT_CONTENT = ["json"]
CELERY_TASK_SERIALIZER = "json"
CELERY_TASK_ACKS_LATE = os.getenv("CELERY_TASK_ACKS_LATE", "true").lower() == "true"
CELERY_TASK_REJECT_ON_WORKER_LOST = os.getenv("CELERY_TASK_REJECT_ON_WORKER_LOST", "true").lower() == "true"
CELERY_BROKER_CONNECTION_RETRY_ON_STARTUP = True
CELERY_TASK_DEFAULT_QUEUE = os.getenv("CELERY_TASK_DEFAULT_QUEUE", "flood-tasks")

# Reliability and observability controls
INLINE_TASK_MAX_RETRIES = int(os.getenv("INLINE_TASK_MAX_RETRIES", "2"))
TASK_DLQ_REDIS_KEY = os.getenv("TASK_DLQ_REDIS_KEY", "flood:dlq")
INGEST_IDEMPOTENCY_TTL_HOURS = int(os.getenv("INGEST_IDEMPOTENCY_TTL_HOURS", "24"))
OPS_ALERT_FAILED_TASKS_PER_HOUR = int(os.getenv("OPS_ALERT_FAILED_TASKS_PER_HOUR", "25"))
OPS_ALERT_DLQ_DEPTH = int(os.getenv("OPS_ALERT_DLQ_DEPTH", "20"))

# Active learning & retraining thresholds
FEEDBACK_RETRAINING_THRESHOLD = int(os.getenv("FEEDBACK_RETRAINING_THRESHOLD", "50"))
FN_SPIKE_THRESHOLD = float(os.getenv("FN_SPIKE_THRESHOLD", "0.02"))  # 2%
RAPID_FEEDBACK_THRESHOLD = int(os.getenv("RAPID_FEEDBACK_THRESHOLD", "20"))  # 20 corrections in 24h
