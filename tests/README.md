# tests/ — Test Suite (37 tests)

## Structure

```
tests/
├── unit/
│   ├── test_fusion.py      — 12 tests: depth calibration logic
│   ├── test_severity.py    — 12 tests: risk classification thresholds
│   ├── test_trigger.py     — 11 tests: MLOps retrain triggers
│   └── test_registry.py    — 12 tests: model version promotion/rollback
└── integration/
    └── test_api.py         — 13 tests: FastAPI endpoint responses
```

## Run all tests
```bash
pytest tests/ -v
```

## Run by category
```bash
pytest tests/unit/ -v          # unit only
pytest tests/integration/ -v   # integration only
pytest tests/ -k "fusion" -v   # single module
```

## Coverage report
```bash
pytest tests/ --cov=pipeline --cov=api --cov-report=term-missing
```

## Adding tests
- Unit tests: pure Python, no DB/network, mock external calls
- Integration tests: spin up `TestClient` from FastAPI, use SQLite in-memory DB
