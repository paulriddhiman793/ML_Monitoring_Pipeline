"""
FastAPI Dashboard — REST API + WebSocket live feed
Access at: http://localhost:8080/docs
"""
import asyncio
import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

app = FastAPI(title="ML Model Monitoring Dashboard", version="1.0.0")
_pipeline = None  # Injected at startup


def set_pipeline(pipeline):
    global _pipeline
    _pipeline = pipeline


# ── Pydantic models ──────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    features: Optional[Dict[str, Any]] = None
    # Dog health convenience fields (either use features dict OR these directly)
    breed: Optional[str] = None
    weight_kg: Optional[float] = None
    heart_rate_bpm: Optional[float] = None
    temperature_celsius: Optional[float] = None
    humidity_pct: Optional[float] = None
    speed_kmh: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

    def resolved_features(self) -> Dict[str, Any]:
        """Return flat feature dict whether user sent nested or flat."""
        if self.features:
            return self.features
        return {k: v for k, v in {
            "breed": self.breed,
            "weight_kg": self.weight_kg,
            "heart_rate_bpm": self.heart_rate_bpm,
            "temperature_celsius": self.temperature_celsius,
            "humidity_pct": self.humidity_pct,
            "speed_kmh": self.speed_kmh,
        }.items() if v is not None}


class GroundTruthRequest(BaseModel):
    prediction_id: str
    actual_value: float


class TrainingRunRequest(BaseModel):
    run_id: Optional[str] = None
    model_type: Optional[str] = "unknown"
    params: Optional[Dict] = {}
    metrics: Optional[Dict] = {}
    feature_importance: Optional[Dict] = {}
    data_stats: Optional[Dict] = {}


# ── Routes ────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def root():
    return """
    <html><body style="font-family:monospace;padding:2rem">
    <h2>🔍 ML Monitoring Pipeline</h2>
    <ul>
      <li><a href="/docs">Swagger UI</a></li>
      <li><a href="/dashboard">Dashboard JSON</a></li>
      <li><a href="/health">Health Check</a></li>
    </ul>
    </body></html>
    """


@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


@app.get("/dashboard")
def dashboard():
    if not _pipeline:
        raise HTTPException(503, "Pipeline not initialized")
    return _pipeline.get_dashboard_data()


@app.post("/predict")
def predict(req: PredictRequest):
    if not _pipeline:
        raise HTTPException(503, "Pipeline not initialized")
    return _pipeline.predict(req.resolved_features(), req.metadata)


@app.post("/ground_truth")
def ground_truth(req: GroundTruthRequest):
    if not _pipeline:
        raise HTTPException(503, "Pipeline not initialized")
    return _pipeline.submit_ground_truth(req.prediction_id, req.actual_value)


@app.post("/training_run")
def register_training_run(req: TrainingRunRequest):
    if not _pipeline:
        raise HTTPException(503, "Pipeline not initialized")
    _pipeline.register_training_run(req.dict())
    return {"success": True}


@app.get("/metrics/performance")
def performance_metrics(hours: int = 24):
    if not _pipeline:
        raise HTTPException(503, "Pipeline not initialized")
    return _pipeline.performance_monitor.calculate_metrics(hours)


@app.get("/metrics/drift")
def drift_summary(days: int = 7):
    if not _pipeline:
        raise HTTPException(503, "Pipeline not initialized")
    return _pipeline.drift_detector.get_drift_summary(_pipeline.db, days)


@app.get("/metrics/quality")
def quality_metrics(hours: int = 24):
    if not _pipeline:
        raise HTTPException(503, "Pipeline not initialized")
    return _pipeline.quality_monitor.get_quality_summary(_pipeline.db, hours)


@app.get("/alerts")
def get_alerts(hours: int = 24):
    if not _pipeline:
        raise HTTPException(503, "Pipeline not initialized")
    return _pipeline.db.get_recent_alerts(hours)


@app.get("/retraining/check")
def check_retraining():
    if not _pipeline:
        raise HTTPException(503, "Pipeline not initialized")
    return _pipeline.retraining_trigger.should_retrain()


@app.post("/run/hourly")
def trigger_hourly():
    if not _pipeline:
        raise HTTPException(503, "Pipeline not initialized")
    _pipeline.run_hourly_checks()
    return {"triggered": "hourly_check"}


@app.post("/run/daily")
def trigger_daily():
    if not _pipeline:
        raise HTTPException(503, "Pipeline not initialized")
    _pipeline.run_daily_evaluation()
    return {"triggered": "daily_evaluation"}


@app.post("/run/weekly")
def trigger_weekly():
    if not _pipeline:
        raise HTTPException(503, "Pipeline not initialized")
    _pipeline.run_weekly_analysis()
    return {"triggered": "weekly_analysis"}


# ── WebSocket live feed ───────────────────────────────────────────────────

class ConnectionManager:
    def __init__(self):
        self.active: List[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        self.active.remove(ws)

    async def broadcast(self, data: Dict):
        msg = json.dumps(data)
        for ws in list(self.active):
            try:
                await ws.send_text(msg)
            except Exception:
                self.active.remove(ws)


manager = ConnectionManager()


@app.websocket("/ws/live")
async def websocket_live(websocket: WebSocket):
    """Real-time prediction stream via WebSocket."""
    await manager.connect(websocket)
    try:
        while True:
            if _pipeline:
                data = _pipeline.get_dashboard_data()
                await websocket.send_text(json.dumps(data))
            await asyncio.sleep(5)
    except WebSocketDisconnect:
        manager.disconnect(websocket)