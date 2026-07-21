# src/observability/ — Metrics + Alerts

## metrics.py — Prometheus
Exposes metrics at `http://localhost:9090/metrics`:

| Metric | Description |
|--------|-------------|
| `flood_predictions_total` | Total predictions (labelled by risk level) |
| `flood_pipeline_latency_seconds` | Per-stage processing time |
| `flood_queue_depth` | Current jobs waiting in queue |
| `flood_model_confidence_avg` | Rolling average model confidence |
| `flood_error_total` | Pipeline errors by stage |

## alerts.py — Slack + SMS
Fires alerts when:
- Risk level = `critical` in any zone
- Model confidence drops below 0.65 (drift detected)
- Pipeline latency > 30 seconds
- Queue depth > 100 jobs

## Setup
```bash
# .env
SLACK_WEBHOOK_URL=https://hooks.slack.com/...
ALERT_PHONE_NUMBER=+91xxxxxxxxxx   # Twilio SMS
```
