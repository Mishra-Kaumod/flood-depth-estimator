# src/api/middleware.py
"""
Security Middleware
====================
1. API Key authentication   — X-API-Key header required on all /predict routes
2. Rate limiting            — per-key sliding window (Redis-backed)
3. CORS                     — configurable allowed origins
4. Request ID               — X-Request-ID injected for tracing
"""

import time
import uuid
import logging
from typing import Optional

from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.cors import CORSMiddleware

log = logging.getLogger("api.middleware")

# Routes that do NOT require an API key
_PUBLIC_PATHS = {"/health", "/docs", "/openapi.json", "/redoc", "/metrics"}


class APIKeyMiddleware(BaseHTTPMiddleware):
    """
    Validates X-API-Key header against configured key.
    Skips public paths. Disabled when api_key_required=False (dev mode only).
    """

    def __init__(self, app, api_key: str, required: bool = True):
        super().__init__(app)
        self.api_key = api_key
        self.required = required
        if not required:
            log.warning("API key auth DISABLED — set API_KEY_REQUIRED=true in production")

    async def dispatch(self, request: Request, call_next):
        # Inject a unique request ID for tracing
        request_id = str(uuid.uuid4())[:8]
        request.state.request_id = request_id

        # Skip auth for public paths
        if not self.required or request.url.path in _PUBLIC_PATHS:
            response = await call_next(request)
            response.headers["X-Request-ID"] = request_id
            return response

        # Check API key
        provided = request.headers.get("X-API-Key", "")
        if not provided:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"error": "Missing X-API-Key header"},
                headers={"X-Request-ID": request_id},
            )
        if provided != self.api_key:
            log.warning("Invalid API key from %s [req=%s]",
                        request.client.host if request.client else "unknown", request_id)
            return JSONResponse(
                status_code=status.HTTP_403_FORBIDDEN,
                content={"error": "Invalid API key"},
                headers={"X-Request-ID": request_id},
            )

        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Sliding window rate limiter per API key.
    Uses Redis if available; falls back to in-memory dict (single-instance only).
    """

    def __init__(self, app, redis_url: str = "", limit: int = 60, window_s: int = 60):
        super().__init__(app)
        self.limit    = limit
        self.window_s = window_s
        self._redis   = None
        self._local: dict[str, list[float]] = {}

        if redis_url:
            try:
                import redis as redis_lib
                client = redis_lib.from_url(redis_url, decode_responses=True)
                client.ping()
                self._redis = client
                log.info("Rate limiter: Redis backend (%d req/%ds)", limit, window_s)
            except Exception:
                log.warning("Rate limiter: Redis unavailable — using in-memory (single node only)")

    async def dispatch(self, request: Request, call_next):
        if request.url.path in _PUBLIC_PATHS:
            return await call_next(request)

        key = request.headers.get("X-API-Key", request.client.host if request.client else "anon")
        allowed = self._check_and_increment(key)

        if not allowed:
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={"error": f"Rate limit exceeded: {self.limit} req/{self.window_s}s"},
                headers={"Retry-After": str(self.window_s)},
            )
        return await call_next(request)

    def _check_and_increment(self, key: str) -> bool:
        now = time.time()
        if self._redis:
            pipe_key = f"ratelimit:{key}"
            with self._redis.pipeline() as pipe:
                pipe.zremrangebyscore(pipe_key, 0, now - self.window_s)
                pipe.zcard(pipe_key)
                pipe.zadd(pipe_key, {str(now): now})
                pipe.expire(pipe_key, self.window_s * 2)
                _, count, *_ = pipe.execute()
            return count < self.limit
        else:
            timestamps = self._local.get(key, [])
            timestamps = [t for t in timestamps if now - t < self.window_s]
            if len(timestamps) >= self.limit:
                return False
            timestamps.append(now)
            self._local[key] = timestamps
            return True


def add_middleware(app, api_key: str, api_key_required: bool,
                   redis_url: str, rate_limit: int, cors_origins: str):
    """Attach all middleware to the FastAPI app. Call before app starts."""
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins.split(","),
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(RateLimitMiddleware, redis_url=redis_url, limit=rate_limit)
    app.add_middleware(APIKeyMiddleware,    api_key=api_key,      required=api_key_required)
