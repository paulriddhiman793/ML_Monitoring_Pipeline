"""
dashboard/app.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FastAPI dashboard — REST API + browser UI at /ui

Endpoints
─────────
Core
  GET  /           → redirect to /ui
  GET  /health
  GET  /dashboard  full snapshot (mode, baseline, metrics, alerts)

Prediction
  POST /predict          all modes — monitored prediction
  POST /ground_truth     submit actual value

Metrics
  GET  /metrics/performance
  GET  /metrics/drift
  GET  /metrics/quality
  GET  /alerts
  GET  /retraining/check

Manual triggers
  POST /run/hourly  /run/daily  /run/weekly

Mode 1 — Auto-simulation (server-side loop)
  POST /simulation/start   {"interval_seconds":5,"max_predictions":0,"auto_ground_truth":true}
  POST /simulation/stop
  GET  /simulation/status

Mode 2 — Continuous frontend feed (browser-side loop)
  POST /feed/start
  POST /feed/stop
  GET  /feed/status
  POST /feed/predict       called by the UI loop on every tick

Mode 3 — Production
  Just use POST /predict normally.

UI
  GET  /ui   serves the monitoring dashboard HTML
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations

import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel

app = FastAPI(
    title="ML Model Monitoring Dashboard",
    version="2.0.0",
    description=(
        "Generic ML monitoring — Mode 1 auto-simulation, "
        "Mode 2 continuous frontend feed, Mode 3 production API."
    ),
)
_pipeline = None   # injected by main.py after pipeline.start()


def set_pipeline(pipeline) -> None:
    global _pipeline
    _pipeline = pipeline


def _p():
    if _pipeline is None:
        raise HTTPException(503, "Pipeline not initialised yet")
    return _pipeline


# ─────────────────────────────────────────────────────────────
# Pydantic models
# ─────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    # Accepts nested {"features": {...}} OR flat dog-health fields
    features:             Optional[Dict[str, Any]] = None
    metadata:             Optional[Dict[str, Any]] = None
    breed:                Optional[str]   = None
    weight_kg:            Optional[float] = None
    heart_rate_bpm:       Optional[float] = None
    temperature_celsius:  Optional[float] = None
    humidity_pct:         Optional[float] = None
    speed_kmh:            Optional[float] = None

    def resolved(self) -> Dict[str, Any]:
        if self.features:
            return self.features
        return {k: v for k, v in {
            "breed":               self.breed,
            "weight_kg":           self.weight_kg,
            "heart_rate_bpm":      self.heart_rate_bpm,
            "temperature_celsius": self.temperature_celsius,
            "humidity_pct":        self.humidity_pct,
            "speed_kmh":           self.speed_kmh,
        }.items() if v is not None}


class GroundTruthRequest(BaseModel):
    prediction_id: str
    actual_value:  float


class TrainingRunRequest(BaseModel):
    run_id:             Optional[str]  = None
    model_type:         Optional[str]  = "unknown"
    params:             Optional[Dict] = {}
    metrics:            Optional[Dict] = {}
    feature_importance: Optional[Dict] = {}
    data_stats:         Optional[Dict] = {}


class SimulationConfig(BaseModel):
    interval_seconds:  float = 5.0
    max_predictions:   int   = 0         # 0 = unlimited
    auto_ground_truth: bool  = True
    feature_overrides: Optional[Dict[str, Any]] = None


class FeedConfig(BaseModel):
    interval_seconds:  float = 3.0
    auto_ground_truth: bool  = False


# ─────────────────────────────────────────────────────────────
# Core
# ─────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/ui")


@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


@app.get("/dashboard")
def dashboard():
    return _p().get_dashboard_data()


# ─────────────────────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────────────────────

@app.post("/predict")
def predict(req: PredictRequest):
    """Standard prediction — used by all three modes and direct Swagger calls."""
    return _p().predict(req.resolved(), req.metadata)


@app.post("/ground_truth")
def ground_truth(req: GroundTruthRequest):
    return _p().submit_ground_truth(req.prediction_id, req.actual_value)


@app.post("/training_run")
def register_training_run(req: TrainingRunRequest):
    _p().register_training_run(req.dict())
    return {"success": True}


# ─────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────

@app.get("/metrics/performance")
def performance(hours: int = 24):
    return _p().performance_monitor.calculate_metrics(hours)


@app.get("/metrics/drift")
def drift(days: int = 7):
    p = _p(); return p.drift_detector.get_drift_summary(p.db, days)


@app.get("/metrics/quality")
def quality(hours: int = 24):
    p = _p(); return p.quality_monitor.get_quality_summary(p.db, hours)


@app.get("/alerts")
def alerts(hours: int = 24):
    return _p().db.get_recent_alerts(hours)


@app.get("/retraining/check")
def retrain_check():
    return _p().retraining_trigger.should_retrain()


# ─────────────────────────────────────────────────────────────
# Manual triggers
# ─────────────────────────────────────────────────────────────

@app.post("/run/hourly")
def run_hourly():
    return _p().run_hourly_checks()


@app.post("/run/daily")
def run_daily():
    return _p().run_daily_evaluation()


@app.post("/run/weekly")
def run_weekly():
    return _p().run_weekly_analysis()


# ─────────────────────────────────────────────────────────────
# Mode 1 — Auto-simulation
# ─────────────────────────────────────────────────────────────

@app.post("/simulation/start")
def start_simulation(cfg: SimulationConfig):
    """
    Start Mode 1 — server-side auto-simulation.
    The pipeline generates random dog-health features and sends them
    to the backend on a background thread at cfg.interval_seconds.
    """
    p        = _p()
    baseline = getattr(p, "baseline_stats", {})
    overrides = cfg.feature_overrides or {}

    def generator() -> Dict[str, Any]:
        features: Dict[str, Any] = {}
        for feat, (lo, hi) in {
            "weight_kg":           [5.0,  45.0],
            "heart_rate_bpm":      [55.0, 170.0],
            "temperature_celsius": [10.0, 40.0],
            "humidity_pct":        [30.0, 95.0],
            "speed_kmh":           [0.0,  20.0],
        }.items():
            if not baseline or feat in baseline:
                rng = overrides.get(feat, [lo, hi])
                features[feat] = round(random.uniform(*rng), 2)

        for feat, stats in baseline.items():
            if stats.get("type") == "categorical":
                cats = stats.get("categories", [])
                dist = stats.get("category_distribution", {})
                if cats:
                    if dist:
                        features[feat] = random.choices(
                            list(dist.keys()), weights=list(dist.values()), k=1)[0]
                    else:
                        features[feat] = random.choice(cats)

        if "breed" not in features and (not baseline or "breed" in baseline):
            features["breed"] = random.choice([
                "Labrador Retriever", "German Shepherd", "Golden Retriever",
                "Bulldog", "Pug", "Beagle", "Poodle", "Rottweiler",
            ])
        return features

    def gt_fn(result: Dict[str, Any]) -> float:
        pred = result.get("prediction", 0.5)
        return round(max(0.0, min(1.0, float(pred) + random.gauss(0, 0.04))), 4)

    p.start_simulation(
        feature_generator=generator,
        interval_seconds=cfg.interval_seconds,
        max_predictions=cfg.max_predictions,
        auto_ground_truth=cfg.auto_ground_truth,
        ground_truth_fn=gt_fn if cfg.auto_ground_truth else None,
    )
    return {
        "started":           True,
        "mode":              1,
        "interval_seconds":  cfg.interval_seconds,
        "max_predictions":   cfg.max_predictions,
        "auto_ground_truth": cfg.auto_ground_truth,
    }


@app.post("/simulation/stop")
def stop_simulation():
    return _p().stop_simulation()


@app.get("/simulation/status")
def simulation_status():
    return _p().simulation_status()


# ─────────────────────────────────────────────────────────────
# Mode 2 — Continuous frontend feed
# ─────────────────────────────────────────────────────────────

@app.post("/feed/start")
def start_feed(cfg: FeedConfig):
    """Arm Mode 2. The browser UI will loop and call POST /feed/predict."""
    result = _p().start_continuous_feed_mode()
    result.update({"interval_seconds": cfg.interval_seconds,
                   "auto_ground_truth": cfg.auto_ground_truth})
    return result


@app.post("/feed/stop")
def stop_feed():
    return _p().stop_continuous_feed_mode()


@app.get("/feed/status")
def feed_status():
    return _p().continuous_feed_status()


@app.post("/feed/predict")
def feed_predict(req: PredictRequest):
    """Called by the browser loop in Mode 2. Same as /predict but increments feed counter."""
    p      = _p()
    result = p.predict(req.resolved(), req.metadata)
    p.record_feed_prediction()
    return result


# ─────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────

@app.get("/ui", response_class=HTMLResponse)
def ui():
    html_path = Path(__file__).parent / "monitoring_ui.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding="utf-8"))
    return HTMLResponse(_fallback_ui())


def _fallback_ui() -> str:
    """Minimal inline dashboard — shown when monitoring_ui.html is missing."""
    return """<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8"/>
<title>ML Monitoring Dashboard</title>
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  body{font-family:'Segoe UI',sans-serif;background:#0d1117;color:#e6edf3;padding:24px}
  h1{color:#58a6ff;margin-bottom:4px;font-size:20px}
  .sub{color:#7d8590;font-size:13px;margin-bottom:20px}
  .grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}
  @media(max-width:700px){.grid{grid-template-columns:1fr}}
  .card{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:18px}
  .card h2{font-size:13px;font-weight:600;color:#8b949e;text-transform:uppercase;
           letter-spacing:.5px;margin-bottom:14px}
  .badge{display:inline-block;padding:2px 10px;border-radius:12px;font-size:11px;
         font-weight:600;margin-left:8px}
  .badge-green{background:#1f4a2a;color:#3fb950}
  .badge-yellow{background:#3d2e00;color:#d29922}
  .badge-red{background:#3d0d0d;color:#f85149}
  .badge-gray{background:#21262d;color:#8b949e}
  label{display:block;font-size:12px;color:#8b949e;margin-bottom:4px;margin-top:10px}
  input,select,textarea{width:100%;background:#1c2128;border:1px solid #30363d;
    color:#e6edf3;padding:8px 10px;border-radius:6px;font-size:13px;font-family:inherit}
  input[type=checkbox]{width:auto}
  .row{display:flex;gap:8px;align-items:center;margin-top:10px}
  .row label{margin:0}
  btn,.btn{display:inline-block;padding:8px 16px;border-radius:6px;border:none;
           cursor:pointer;font-size:13px;font-weight:500}
  .btn-green{background:#238636;color:#fff}
  .btn-red{background:#da3633;color:#fff}
  .btn-blue{background:#1f6feb;color:#fff}
  .btn-gray{background:#21262d;color:#e6edf3;border:1px solid #30363d}
  .btn:disabled{opacity:.4;cursor:not-allowed}
  pre{background:#1c2128;border:1px solid #30363d;border-radius:6px;padding:12px;
      font-size:11px;overflow:auto;max-height:220px;margin-top:10px;white-space:pre-wrap}
  .stat-row{display:flex;justify-content:space-between;padding:6px 0;
            border-bottom:1px solid #21262d;font-size:13px}
  .stat-row:last-child{border-bottom:none}
  .stat-val{color:#58a6ff;font-weight:600}
  .sim-status{padding:8px 12px;border-radius:6px;background:#1c2128;
              border:1px solid #30363d;font-size:12px;margin-top:10px}
</style>
</head>
<body>
<h1>🐕 ML Monitoring Dashboard</h1>
<p class="sub">Real-time monitoring · 3 modes · dog_health adapter</p>

<div class="grid">

  <!-- STATUS CARD -->
  <div class="card">
    <h2>Pipeline Status <span id="modeBadge" class="badge badge-gray">–</span></h2>
    <div id="statusRows"><div style="color:#7d8590">Loading…</div></div>
    <button class="btn btn-gray" style="margin-top:12px" onclick="loadDashboard()">↻ Refresh</button>
  </div>

  <!-- MODE 1 — AUTO SIMULATION -->
  <div class="card">
    <h2>Mode 1 — Auto-Simulation <span id="simBadge" class="badge badge-gray">Stopped</span></h2>
    <p style="font-size:12px;color:#7d8590;margin-bottom:8px">
      Pipeline generates synthetic dog-health features and fires them at your
      backend automatically on a timer.
    </p>
    <label>Interval (seconds)</label>
    <input id="simInterval" type="number" value="5" min="1" max="60"/>
    <label>Max predictions (0 = unlimited)</label>
    <input id="simMax" type="number" value="0" min="0"/>
    <div class="row">
      <input type="checkbox" id="simAutoGT" checked/>
      <label>Auto-submit ground truth (enables RMSE / R²)</label>
    </div>
    <div style="display:flex;gap:8px;margin-top:12px">
      <button class="btn btn-green" id="simStart" onclick="startSim()" style="flex:1">▶ Start</button>
      <button class="btn btn-red"   id="simStop"  onclick="stopSim()" disabled>■ Stop</button>
    </div>
    <div class="sim-status" id="simInfo">Not running</div>
  </div>

  <!-- MODE 2 — CONTINUOUS FEED -->
  <div class="card">
    <h2>Mode 2 — Continuous Feed <span id="feedBadge" class="badge badge-gray">Stopped</span></h2>
    <p style="font-size:12px;color:#7d8590;margin-bottom:8px">
      Fill the form below and this UI loops, firing real predictions to your
      backend every N seconds until you stop it.
    </p>
    <label>Interval (seconds)</label>
    <input id="feedInterval" type="number" value="3" min="1" max="60"/>
    <div style="display:flex;gap:8px;margin-top:12px">
      <button class="btn btn-blue" id="feedStart" onclick="startFeed()" style="flex:1">▶ Start Feed</button>
      <button class="btn btn-red"  id="feedStop"  onclick="stopFeed()" disabled>■ Stop</button>
    </div>
    <div class="sim-status" id="feedInfo">Not running</div>
  </div>

  <!-- MODE 3 / MANUAL PREDICT -->
  <div class="card">
    <h2>Mode 3 — Manual / Production Predict</h2>
    <p style="font-size:12px;color:#7d8590;margin-bottom:8px">
      Send one prediction now. In production your app calls POST /predict directly.
    </p>
    <label>Features (JSON)</label>
    <textarea id="featJson" rows="7">{
  "breed": "Labrador Retriever",
  "weight_kg": 30,
  "heart_rate_bpm": 85,
  "temperature_celsius": 25,
  "humidity_pct": 60,
  "speed_kmh": 5.5
}</textarea>
    <button class="btn btn-green" style="margin-top:10px;width:100%" onclick="manualPredict()">
      🐕 Send Prediction
    </button>
    <pre id="predResult" style="display:none"></pre>
  </div>

  <!-- GROUND TRUTH -->
  <div class="card">
    <h2>Submit Ground Truth</h2>
    <label>Prediction ID</label>
    <input id="gtId" placeholder="pred_20260304_xxxxxxxx"/>
    <label>Actual value</label>
    <input id="gtVal" type="number" step="0.001" value="0.15"/>
    <button class="btn btn-blue" style="margin-top:10px;width:100%" onclick="submitGT()">
      Submit Ground Truth
    </button>
    <pre id="gtResult" style="display:none"></pre>
  </div>

  <!-- METRICS -->
  <div class="card">
    <h2>Metrics</h2>
    <div style="display:flex;flex-wrap:wrap;gap:6px;margin-bottom:10px">
      <button class="btn btn-gray" onclick="loadMetric('/metrics/performance')">Performance</button>
      <button class="btn btn-gray" onclick="loadMetric('/metrics/drift')">Drift</button>
      <button class="btn btn-gray" onclick="loadMetric('/metrics/quality')">Quality</button>
      <button class="btn btn-gray" onclick="loadMetric('/alerts')">Alerts</button>
      <button class="btn btn-gray" onclick="loadMetric('/retraining/check')">Retrain?</button>
    </div>
    <pre id="metricsResult">Select a metric above</pre>
  </div>

  <!-- EVALUATIONS -->
  <div class="card">
    <h2>Manual Evaluations</h2>
    <p style="font-size:12px;color:#7d8590;margin-bottom:12px">
      Trigger evaluation jobs that normally run on a schedule.
    </p>
    <div style="display:flex;gap:8px;flex-wrap:wrap">
      <button class="btn btn-gray" onclick="runEval('/run/hourly')">Hourly</button>
      <button class="btn btn-gray" onclick="runEval('/run/daily')">Daily</button>
      <button class="btn btn-gray" onclick="runEval('/run/weekly')">Weekly</button>
    </div>
    <pre id="evalResult" style="display:none"></pre>
  </div>

</div>

<p style="margin-top:20px;font-size:12px;color:#444">
  <a href="/docs" style="color:#58a6ff">Swagger UI</a> ·
  Auto-refresh every 5 s
</p>

<script>
const API = window.location.origin;
let feedTimer = null;

// ── Helpers ─────────────────────────────────────────────────
async function get(url) {
  const r = await fetch(API + url);
  return r.json();
}
async function post(url, body) {
  const r = await fetch(API + url, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(body),
  });
  return r.json();
}

// ── Dashboard ────────────────────────────────────────────────
async function loadDashboard() {
  try {
    const d = await get('/dashboard');
    const rows = document.getElementById('statusRows');

    const modeLabel = d.mode_label || '–';
    const modeBadge = document.getElementById('modeBadge');
    modeBadge.textContent = modeLabel;
    modeBadge.className = 'badge ' + ({
      'Auto-Simulation': 'badge-green',
      'Continuous Feed': 'badge-yellow',
      'Production':      'badge-green',
    }[modeLabel] || 'badge-gray');

    const preds = d.predictions_last_24h || 0;
    const perf  = d.performance || {};
    const qual  = d.data_quality || {};

    rows.innerHTML = [
      ['Model',          d.model_version || '–'],
      ['Adapter',        d.adapter_type  || '–'],
      ['Predictions 24h', preds],
      ['RMSE',           perf.rmse != null ? perf.rmse.toFixed(4) : 'N/A'],
      ['R²',             perf.r2   != null ? perf.r2.toFixed(4)   : 'N/A'],
      ['Quality rate',   qual.data_quality_rate != null
                          ? (qual.data_quality_rate * 100).toFixed(1) + '%' : 'N/A'],
      ['Baseline',       d.baseline_ready ? '✓ Ready' : '⏳ Collecting…'],
      ['Alerts 24h',     (d.recent_alerts || []).length],
    ].map(([k, v]) =>
      `<div class="stat-row"><span>${k}</span><span class="stat-val">${v}</span></div>`
    ).join('');

    // Sync sim badge
    const sim = d.simulation_status || {};
    setSimUI(sim.running, sim.predictions_sent);

    // Sync feed badge
    const feed = d.feed_status || {};
    setFeedUI(feed.armed, feed.predictions_sent);

  } catch(e) {
    document.getElementById('statusRows').innerHTML =
      '<div style="color:#f85149">Could not load dashboard</div>';
  }
}

// ── Mode 1 — Simulation ──────────────────────────────────────
async function startSim() {
  const d = await post('/simulation/start', {
    interval_seconds:  parseFloat(document.getElementById('simInterval').value) || 5,
    max_predictions:   parseInt(document.getElementById('simMax').value)        || 0,
    auto_ground_truth: document.getElementById('simAutoGT').checked,
  });
  if (d.started) setSimUI(true, 0);
  document.getElementById('simInfo').textContent =
    `Started — every ${d.interval_seconds}s` +
    (d.max_predictions ? `, max ${d.max_predictions}` : ', unlimited') +
    (d.auto_ground_truth ? ', auto-GT on' : '');
}

async function stopSim() {
  const d = await post('/simulation/stop', {});
  setSimUI(false, d.total_predictions_sent);
  document.getElementById('simInfo').textContent =
    `Stopped — ${d.total_predictions_sent} predictions sent`;
}

function setSimUI(running, count) {
  const badge = document.getElementById('simBadge');
  const start = document.getElementById('simStart');
  const stop  = document.getElementById('simStop');
  if (running) {
    badge.textContent = '● Live';
    badge.className   = 'badge badge-green';
    start.disabled    = true;
    stop.disabled     = false;
    if (count != null)
      document.getElementById('simInfo').textContent = `Running — ${count} sent so far`;
  } else {
    badge.textContent = 'Stopped';
    badge.className   = 'badge badge-gray';
    start.disabled    = false;
    stop.disabled     = true;
  }
}

// ── Mode 2 — Feed ────────────────────────────────────────────
async function startFeed() {
  const interval = parseFloat(document.getElementById('feedInterval').value) || 3;
  await post('/feed/start', {interval_seconds: interval});
  setFeedUI(true, 0);

  feedTimer = setInterval(async () => {
    try {
      const features = JSON.parse(document.getElementById('featJson').value);
      const r = await post('/feed/predict', {features});
      const cnt = (await get('/feed/status')).predictions_sent || 0;
      document.getElementById('feedInfo').textContent =
        `Running — ${cnt} predictions sent  (last pred: ${r.prediction ?? 'err'})`;
      setFeedUI(true, cnt);
      if (r.prediction_id) document.getElementById('gtId').value = r.prediction_id;
    } catch(e) {
      document.getElementById('feedInfo').textContent = 'Error: ' + e.message;
    }
  }, interval * 1000);
}

async function stopFeed() {
  clearInterval(feedTimer);
  feedTimer = null;
  const d = await post('/feed/stop', {});
  setFeedUI(false, d.predictions_sent);
  document.getElementById('feedInfo').textContent =
    `Stopped — ${d.predictions_sent} predictions sent`;
}

function setFeedUI(armed, count) {
  const badge = document.getElementById('feedBadge');
  const start = document.getElementById('feedStart');
  const stop  = document.getElementById('feedStop');
  if (armed) {
    badge.textContent = '● Live';
    badge.className   = 'badge badge-yellow';
    start.disabled    = true;
    stop.disabled     = false;
  } else {
    badge.textContent = 'Stopped';
    badge.className   = 'badge badge-gray';
    start.disabled    = false;
    stop.disabled     = true;
  }
}

// ── Mode 3 — Manual predict ──────────────────────────────────
async function manualPredict() {
  try {
    const features = JSON.parse(document.getElementById('featJson').value);
    const r = await post('/predict', {features});
    const el = document.getElementById('predResult');
    el.style.display = 'block';
    el.textContent = JSON.stringify(r, null, 2);
    if (r.prediction_id) document.getElementById('gtId').value = r.prediction_id;
  } catch(e) {
    document.getElementById('predResult').textContent = 'Error: ' + e.message;
    document.getElementById('predResult').style.display = 'block';
  }
}

// ── Ground truth ─────────────────────────────────────────────
async function submitGT() {
  const r = await post('/ground_truth', {
    prediction_id: document.getElementById('gtId').value,
    actual_value:  parseFloat(document.getElementById('gtVal').value),
  });
  const el = document.getElementById('gtResult');
  el.style.display = 'block';
  el.textContent = JSON.stringify(r, null, 2);
}

// ── Metrics ──────────────────────────────────────────────────
async function loadMetric(url) {
  const d  = await get(url);
  const el = document.getElementById('metricsResult');
  el.textContent = JSON.stringify(d, null, 2);
}

// ── Evaluations ──────────────────────────────────────────────
async function runEval(url) {
  const d  = await post(url, {});
  const el = document.getElementById('evalResult');
  el.style.display = 'block';
  el.textContent = JSON.stringify(d, null, 2);
}

// ── Auto-refresh ─────────────────────────────────────────────
loadDashboard();
setInterval(loadDashboard, 5000);
</script>
</body>
</html>"""