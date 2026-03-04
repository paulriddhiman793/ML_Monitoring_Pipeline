# 🐕 ML Model Monitoring Pipeline

A production-grade, generic ML monitoring system that connects to any model backend API and continuously tracks prediction quality, data drift, performance degradation, and retraining signals — all with zero changes to your existing model code.

Built around 7 MLOps monitoring components, an adapter pattern for plugging in any backend, and 3 operating modes so you can test, demo, or deploy with equal ease.

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Operating Modes](#operating-modes)
- [Adapter System](#adapter-system)
- [API Reference](#api-reference)
- [Configuration](#configuration)
- [Dashboard UI](#dashboard-ui)
- [Monitoring Components](#monitoring-components)
- [Extending the Pipeline](#extending-the-pipeline)
- [Troubleshooting](#troubleshooting)

---

## Features

- **7 MLOps monitoring components** — logging, data quality, performance, drift detection, explainability, retraining triggers, and alerting
- **3 operating modes** — auto-simulation for testing, continuous frontend feed for demos, production API for real deployments
- **Adapter pattern** — 6 built-in adapters for any backend response format; write a custom adapter in ~10 lines
- **Zero-config baseline** — if no baseline is provided, the pipeline auto-infers it from the first 50 predictions
- **Interactive mode selection** — after starting, a CLI prompt asks how you want to run
- **Live dashboard UI** — built-in browser interface at `/ui` with real-time charts and mode controls
- **Swagger UI** — full REST API docs at `/docs`
- **SQLite persistence** — all predictions, metrics, drift logs, and alerts stored locally

---

## Architecture

```
Your App / Simulation / Browser UI
         │
         ▼
   PredictionProxy          ← intercepts every prediction
         │
         ├── DataQualityMonitor   validate input features
         ├── MLMonitoringLogger   persist to SQLite
         │
         ▼
   MLBackendClient  ──[ Adapter ]──►  Your ML Backend API
         │
         └── returns: prediction_id, prediction, inference_time_ms
                                │
                    Background Scheduler
                         │
                         ├── Hourly:  quality rate · latency · alerts
                         ├── Daily:   performance (RMSE/R²) · drift · alerts
                         └── Weekly:  SHAP · importance · retraining decision
```

### Adapter Pattern

All backend-specific knowledge lives in one class. The rest of the pipeline never sees field names like `risk_score` or `data.prediction`.

```
Backend response                    Adapter                   Pipeline internal
────────────────                    ───────                   ─────────────────
{"status":"ok",                     NestedAdapter             prediction: 0.116
 "data":{"risk_score":0.116}}  ──►  data_key="data"      ──► extra: {status:ok}
                                    prediction_field=
                                    "risk_score"
```

---

## Project Structure

```
ml-monitoring-pipeline/
├── main.py                    ← entry point + interactive mode selection
├── requirements.txt
├── .env.example
├── config/
│   └── config.yaml            ← thresholds, schedule, adapter config
│
├── api/
│   ├── __init__.py
│   ├── adapters.py            ← 6 built-in adapters + registry + factory
│   ├── client.py              ← HTTP client (adapter-driven, domain-agnostic)
│   └── proxy.py               ← prediction interceptor
│
├── core/
│   ├── __init__.py
│   ├── logger.py              ← Component 1: logging layer
│   ├── data_quality.py        ← Component 2: input validation
│   ├── performance.py         ← Component 3: RMSE · MAE · R² · bias
│   ├── drift_detector.py      ← Component 4: PSI · KS · chi²
│   ├── explainability.py      ← Component 5: feature importance · SHAP
│   ├── retraining.py          ← Component 6: multi-signal retraining trigger
│   └── alerts.py              ← Component 7: console / Slack / email alerts
│
├── db/
│   ├── __init__.py
│   ├── database.py            ← synchronous SQLite wrapper
│   └── schema.sql             ← all table definitions
│
├── pipeline/
│   ├── __init__.py
│   └── orchestrator.py        ← wires all components, owns 3 modes
│
└── dashboard/
    ├── __init__.py
    ├── app.py                 ← FastAPI REST API + /ui endpoint
    └── monitoring_ui.html     ← browser dashboard (optional, has inline fallback)
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env — set ML_BACKEND_URL at minimum
```

### 3. Start the pipeline

```bash
# Dog health backend (the example backend this was built and tested with)
python main.py start \
  --backend-url https://rppooo-dog-backend.hf.space \
  --adapter dog_health \
  --model-info-endpoint /model-info

# Generic flat backend: {"prediction": 42.3}
python main.py start \
  --backend-url http://my-api:8000 \
  --adapter flat \
  --prediction-field prediction

# Skip the interactive prompt, jump straight to a mode
python main.py start --backend-url https://... --adapter dog_health --mode 1
```

### 4. Choose a mode

After startup the CLI asks:

```
╭──────────────────────────────────────────────────────────╮
│   ML Monitoring Pipeline — Select Mode                 │
│                                                          │
│  1  Auto-Simulation     pipeline generates predictions   │
│  2  Continuous Feed     browser UI loops automatically   │
│  3  Production / API    your app calls POST /predict     │
╰──────────────────────────────────────────────────────────╯
Enter mode number [1/2/3] (3):
```

### 5. Open the dashboard

- **Swagger UI** → `http://localhost:8080/docs`
- **Browser UI** → `http://localhost:8080/ui`

---

## Operating Modes

### Mode 1 — Auto-Simulation

The pipeline generates synthetic feature values on a background timer and sends them to your backend automatically. Nothing else is needed from you — just watch the monitoring data accumulate.

**Best for:** testing all 7 monitoring components, CI/CD validation, demos without real traffic.

**How it starts:**

```
Enter mode number: 1

Seconds between predictions [5]:  3
Max predictions (0 = run forever) [0]:  100
Auto-submit ground truth? [Y/n]:  Y
Customise feature ranges? [y/N]:  N

▶ Mode 1 — Auto-Simulation is running
  Interval  : 3s
  Max preds : 100
  Auto-GT   : yes
```

You can also start/stop simulation at runtime via the API:

```bash
# Start
curl -X POST http://localhost:8080/simulation/start \
  -H "Content-Type: application/json" \
  -d '{"interval_seconds": 3, "max_predictions": 50, "auto_ground_truth": true}'

# Stop
curl -X POST http://localhost:8080/simulation/stop

# Status
curl http://localhost:8080/simulation/status
```

---

### Mode 2 — Continuous Frontend Feed

The browser dashboard UI loops and fires real predictions at a set interval. You control start/stop from the browser — no terminal interaction needed during the run.

**Best for:** interactive demos, load testing with real inputs, showing stakeholders live charts filling up.

**How it works:**

1. Start the pipeline and choose mode `2`
2. Open `http://localhost:8080/ui`
3. Fill in the feature form (or leave the defaults)
4. Set the interval (seconds) in the **Continuous Feed** section
5. Click **▶ Start Continuous Feed**
6. Watch charts update every N seconds
7. Click **■ Stop** when done

---

### Mode 3 — Production / API

The pipeline starts silently and waits. Your application calls `POST /predict` on every real model request. This is the intended production use — the monitoring layer is invisible to your users.

**Best for:** live production deployments, real user traffic, connecting to your existing application.

**Integration example:**

```python
import httpx

# Your app — just add these two calls
response = httpx.post("http://localhost:8080/predict", json={
    "breed": "Labrador Retriever",
    "weight_kg": 30.0,
    "heart_rate_bpm": 85.0,
    "temperature_celsius": 25.0,
    "humidity_pct": 60.0,
    "speed_kmh": 5.5,
})
result = response.json()
prediction_id = result["prediction_id"]

# Later, when you know the real outcome:
httpx.post("http://localhost:8080/ground_truth", json={
    "prediction_id": prediction_id,
    "actual_value": 0.14,
})
```

---

## Adapter System

The adapter is the only part of the pipeline that knows what your backend's response looks like. Swap adapters to use any ML backend without changing the monitoring code.

### Built-in adapters

| Adapter | `--adapter` flag | Use when |
|---|---|---|
| `FlatPredictionAdapter` | `flat` | `{"prediction": 42.3}` |
| `NestedPredictionAdapter` | `nested` | `{"status":"ok","data":{"value":42.3}}` |
| `WrappedFeaturesAdapter` | `wrapped` | Backend expects `{"features":{...}}` in request |
| `ClassificationAdapter` | `classification` | `{"label":"fraud","confidence":0.91}` |
| `ListOutputAdapter` | `list` | `{"predictions":[42.3]}` |
| `DogHealthAdapter` | `dog_health` | The example dog health backend |

### Choosing an adapter

```
Backend returns {"prediction": 42.3}
  → --adapter flat --prediction-field prediction

Backend returns {"score": 0.87}
  → --adapter flat --prediction-field score

Backend returns {"status":"ok", "data":{"value":42.3}}
  → --adapter nested --data-key data --prediction-field value

Backend returns {"label":"fraud", "confidence":0.91}
  → --adapter classification

Backend returns {"predictions":[42.3]}
  → --adapter list

Backend expects {"features":{...}} in request body
  → --adapter wrapped

Your dog health backend
  → --adapter dog_health --model-info-endpoint /model-info
```

### Writing a custom adapter

```python
# api/adapters.py — add to the bottom

from api.adapters import BaseAdapter, ADAPTER_REGISTRY
from typing import Any, Dict

class MyBackendAdapter(BaseAdapter):
    def build_request_body(self, features: Dict[str, Any]) -> Dict[str, Any]:
        # Transform features into whatever shape your backend expects
        return {"input_data": features, "version": "v2"}

    def extract_prediction(self, response: Dict[str, Any]) -> float:
        # Pull the numeric prediction value out of the response
        return float(response["payload"]["results"][0]["pred"])

    def extract_extra(self, response: Dict[str, Any]) -> Dict[str, Any]:
        # Any additional fields to store alongside the prediction
        return {"confidence": response["payload"]["results"][0].get("conf")}

# Register it
ADAPTER_REGISTRY["my_backend"] = MyBackendAdapter
```

Then use it:

```bash
python main.py start --backend-url http://my-api:8000 --adapter my_backend
```

---

## API Reference

All endpoints are also documented interactively at `http://localhost:8080/docs`.

### Core

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Liveness check |
| `GET` | `/dashboard` | Full snapshot — mode, baseline, all metrics, alerts |

### Prediction

| Method | Endpoint | Body | Description |
|---|---|---|---|
| `POST` | `/predict` | `{"features":{...}}` | Monitored prediction (all modes) |
| `POST` | `/ground_truth` | `{"prediction_id":"pred_...","actual_value":0.14}` | Log real outcome |
| `POST` | `/training_run` | training metadata | Register a new training run |

### Metrics

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/metrics/performance` | RMSE · MAE · R² · bias · percentiles vs baseline |
| `GET` | `/metrics/drift` | PSI · KS statistic · chi² per feature |
| `GET` | `/metrics/quality` | Data quality rate · issue breakdown |
| `GET` | `/alerts` | Recent alerts (last 24h by default) |
| `GET` | `/retraining/check` | Retraining decision with confidence score and reasons |

### Manual evaluation triggers

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/run/hourly` | Quality + latency snapshot |
| `POST` | `/run/daily` | Performance + drift + alerts |
| `POST` | `/run/weekly` | SHAP + feature importance + retraining decision |

### Mode 1 — Simulation

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/simulation/start` | Start auto-simulation |
| `POST` | `/simulation/stop` | Stop auto-simulation |
| `GET` | `/simulation/status` | Running state + predictions sent |

`POST /simulation/start` body:
```json
{
  "interval_seconds": 5,
  "max_predictions": 0,
  "auto_ground_truth": true,
  "feature_overrides": {
    "weight_kg": [10, 40],
    "heart_rate_bpm": [60, 140]
  }
}
```

### Mode 2 — Continuous feed

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/feed/start` | Arm continuous feed mode |
| `POST` | `/feed/stop` | Stop continuous feed |
| `GET` | `/feed/status` | Armed state + predictions sent |
| `POST` | `/feed/predict` | Called by browser UI loop on each tick |

---

## Configuration

### `.env`

```env
ML_BACKEND_URL=https://rppooo-dog-backend.hf.space
ML_API_KEY=
DATABASE_URL=monitoring.db
DASHBOARD_HOST=0.0.0.0
DASHBOARD_PORT=8080
```

### `config/config.yaml`

```yaml
# Adapter config (can also be set via CLI flags)
adapter:
  type: dog_health          # flat | nested | wrapped | classification | list | dog_health
  prediction_field: prediction
  data_key: data            # nested adapter only

# Provide baseline upfront (skips the 50-prediction bootstrap wait)
baseline_stats:
  weight_kg:
    type: numerical
    mean: 25.0
    std: 10.0
    min: 5.0
    max: 45.0
    missing_rate: 0.0
  breed:
    type: categorical
    categories: [Labrador Retriever, Pug, German Shepherd]
    category_distribution:
      Labrador Retriever: 0.4
      Pug: 0.3
      German Shepherd: 0.3
    missing_rate: 0.0

# Custom business-rule validators
validation_rules:
  - name: weight_positive
    description: Dog weight must be > 0
    condition: "lambda f: f.get('weight_kg', 1) > 0"
  - name: hr_in_range
    description: Heart rate must be 30–300
    condition: "lambda f: 30 <= f.get('heart_rate_bpm', 100) <= 300"

# Monitoring schedule
monitoring_interval_seconds: 3600
daily_eval_hour: 2
weekly_eval_day: 0            # 0 = Monday

# Drift / performance / alert thresholds
thresholds:
  drift:
    psi_threshold: 0.2
    ks_statistic_threshold: 0.1
    chi2_pvalue_threshold: 0.05
    mean_shift_threshold: 0.15
    std_shift_threshold: 0.20
  retraining:
    performance_degradation_pct: 10
    min_days_since_training: 30
    min_new_samples: 10000
  alerts:
    rmse_increase_pct: 15
    r2_decrease: 0.05
    drift_severities: [high, critical]
    data_quality_rate: 0.90

# Alert channels: console | email | slack
alert_channels:
  - console
```

### CLI flags

```
--backend-url           ML backend API URL  (overrides .env)
--adapter               Adapter type: flat | nested | wrapped | classification | list | dog_health
--prediction-field      Response field containing the numeric prediction
--data-key              Outer wrapper key (nested adapter only)
--predict-endpoint      Path for predictions  (default: /predict)
--model-info-endpoint   Path for model info   (default: /model/info)
--health-endpoint       Path for health check (default: /health)
--mode                  Skip prompt: 1=simulation  2=continuous  3=production
--host                  Dashboard host (default: 0.0.0.0)
--port                  Dashboard port (default: 8080)
--config                Config file path (default: config/config.yaml)
```

---

## Dashboard UI

Open `http://localhost:8080/ui` for the browser dashboard. It provides:

- **Pipeline Status card** — model version, adapter, predictions in last 24h, RMSE, R², quality rate, baseline readiness, active alerts
- **Mode 1 card** — set interval and max, click ▶ Start to begin auto-simulation, watch the Live badge
- **Mode 2 card** — arm the feed, edit the feature form, browser fires predictions in a loop
- **Mode 3 card** — send a single manual prediction, copy the `prediction_id`
- **Ground Truth card** — paste a `prediction_id`, enter the actual value, submit
- **Metrics buttons** — one click to fetch performance / drift / quality / alerts / retrain decision
- **Manual Evaluation** — trigger hourly / daily / weekly jobs on demand

The dashboard auto-refreshes every 5 seconds.

If `dashboard/monitoring_ui.html` is missing the pipeline serves a minimal inline fallback UI that covers all the same functions.

---

## Monitoring Components

### Component 1 — Logging

Every prediction is logged to SQLite with: `prediction_id`, timestamp, model version, input features (JSON), prediction value, inference latency, quality flags, and optional metadata. Ground truth is linked by `prediction_id`.

### Component 2 — Data Quality Monitor

Validates each prediction input against the baseline:
- Missing required features → blocked (critical)
- Unexpected null / NaN values → blocked (high)
- Out-of-range values (10% buffer past training min/max) → warning (medium)
- Extreme z-scores (> 3σ) → warning
- Unseen categorical values → warning
- Custom business rules → configurable severity

### Component 3 — Performance Monitor

Calculates metrics against logged ground truth:
- RMSE, MAE, MAPE, R², bias, error std, max error, median absolute error
- Percentiles: p10, p50, p90, p95, p99
- Comparison to training baseline (% change in RMSE / MAE, R² delta)
- Error segment analysis by feature quartile and category

### Component 4 — Drift Detector

Compares current feature distributions to baseline:
- **Numerical:** PSI (Population Stability Index), KS statistic, mean shift %, std shift %
- **Categorical:** PSI, chi-squared test, new/missing categories
- Overall drift severity: none / low / medium / high / critical
- Weekly drift summary with timeline

### Component 5 — Explainability Monitor

- CatBoost native feature importance (requires model object in config)
- SHAP values on recent prediction samples
- Importance change % vs training baseline
- Anomalous prediction detection (> 95th percentile SHAP magnitude)

### Component 6 — Retraining Trigger

Multi-signal decision engine:
1. Performance degradation (RMSE increase %)
2. Data drift severity (high / critical)
3. Time since last training (configurable threshold)
4. Sufficient new labeled samples (configurable minimum)
5. High-error feature segments

Outputs: `should_retrain`, `confidence` (low / medium / high), `reasons` list, `blocked_reason` if data is insufficient.

### Component 7 — Alert System

Fires alerts when thresholds are crossed:
- RMSE increase > threshold
- R² decrease > threshold
- Drift severity in [high, critical]
- Data quality rate < threshold
- Prediction latency > threshold

Default channel: `console`. Extend to Slack or email in `core/alerts.py`.

---

## Extending the Pipeline

### Adding a new alert channel

In `core/alerts.py`, add a handler in `_dispatch()`:

```python
def _dispatch(self, alert: Dict) -> None:
    if "console" in self.channels:
        # existing console code
        ...
    if "slack" in self.channels:
        import httpx
        httpx.post(os.getenv("SLACK_WEBHOOK_URL"), json={
            "text": f"[{alert['severity'].upper()}] {alert['type']}: {alert['message']}"
        })
    if "email" in self.channels:
        # your SMTP code here
        pass
```

Then add `slack` or `email` to `alert_channels` in your config.

### Hot-swapping the baseline

```python
# Via Python
pipeline.update_baseline({
    "weight_kg": {"type":"numerical","mean":28.0,"std":11.0,"min":5.0,"max":45.0,"missing_rate":0.0},
    ...
})

# Inspect current baseline
print(pipeline.get_baseline())
```

### Registering a new training run

```bash
curl -X POST http://localhost:8080/training_run \
  -H "Content-Type: application/json" \
  -d '{
    "run_id": "run_20260304_v4",
    "model_type": "CatBoost",
    "metrics": {"rmse": 0.041, "mae": 0.028, "r2": 0.94},
    "feature_importance": {"weight_kg": 0.35, "heart_rate_bpm": 0.28},
    "data_stats": {
      "weight_kg": {"type":"numerical","mean":25.0,"std":10.0,"min":5.0,"max":45.0,"missing_rate":0.0}
    }
  }'
```

This updates the baseline, refreshes drift detection, and reseeds performance comparisons.

### Adding a prediction callback (e.g. WebSocket push)

```python
def my_callback(result: dict) -> None:
    # Called after every successful prediction
    websocket_manager.broadcast(result)

pipeline.add_prediction_callback(my_callback)
```

---

## Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `TypeError: DogHealthAdapter.__init__() got unexpected keyword argument 'prediction_field'` | Old `get_adapter` passes all CLI flags including unsupported ones | The new `get_adapter` filters kwargs via `inspect.signature` — make sure you're using the latest `api/adapters.py` |
| `ModuleNotFoundError: api.adapters` | Missing file | Create `api/adapters.py` from the adapter system code |
| `Could not extract prediction from response` | Wrong adapter type or field name | Check your backend's actual response shape and choose the right `--adapter` and `--prediction-field` |
| `Baseline not ready` on drift checks | Not enough predictions accumulated yet | Either provide `baseline_stats` in config or wait for 50+ predictions |
| `No ground truth data` on performance | Haven't called `/ground_truth` yet | Post actual values via `POST /ground_truth` after each prediction |
| `Connection refused` on backend | HuggingFace space is sleeping | Open the space URL in a browser to wake it, wait ~30 seconds, retry |
| `Got unexpected extra arguments` | Backslashes copied as literal `\` characters on Windows | Run as a single line with no line breaks, or use PowerShell backtick (`` ` ``) for continuation |
| `apscheduler` errors | Missing dependency | `pip install apscheduler` |
| Drift always shows `none` | Baseline not ready or not enough recent predictions | Run simulation for a while first, then trigger `POST /run/daily` |

---

## Requirements

```
fastapi>=0.111.0
uvicorn>=0.29.0
httpx>=0.27.0
numpy>=1.26.4
pandas>=2.2.2
scipy>=1.13.0
scikit-learn>=1.4.2
catboost>=1.2.5
shap>=0.45.0
sqlalchemy>=2.0.30
aiosqlite>=0.20.0
pyyaml>=6.0.1
python-dotenv>=1.0.1
apscheduler>=3.10.4
rich>=13.7.1
typer>=0.12.3
```

---

## License

MIT
