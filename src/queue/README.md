# src/queue/ — Job Queue (Redis + Retry + Dead-Letter)

Handles image processing jobs reliably — retries on failure, dead-letter queue for failed jobs.

## How it works
```
Image uploaded → job_queue.enqueue(image_path)
                      ↓
              Worker picks up job
                      ↓
         Success → write to DB
         Failure → retry (max 3x)
         All retries fail → dead-letter queue (DLQ)
```

## Dead-Letter Queue (DLQ)
Failed jobs land in Redis list `flood:dlq:events`.  
Review them:
```bash
redis-cli LRANGE flood:dlq:events 0 -1
```

## Configuration (config.yaml)
```yaml
event_processing:
  retry:
    max_attempts: 3
    base_delay_seconds: 0.5
  dlq:
    backend: redis
    redis_url: redis://localhost:6379/2
```
