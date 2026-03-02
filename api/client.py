"""
ML Backend Client — Dog Health Monitoring API
Backend: https://rppooo-dog-backend.hf.space
"""
import time
import httpx
from typing import Any, Dict, Optional
from rich.console import Console

console = Console()


class MLBackendClient:

    def __init__(self, base_url: str, api_key: Optional[str] = None,
                 predict_endpoint: str = "/predict",
                 model_info_endpoint: str = "/model-info"):
        self.base_url = base_url.rstrip("/")
        self.predict_endpoint = predict_endpoint
        self.model_info_endpoint = model_info_endpoint
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"

    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        POST /predict
        Body:  { "breed": ..., "weight_kg": ..., "heart_rate_bpm": ...,
                 "temperature_celsius": ..., "humidity_pct": ..., "speed_kmh": ... }
        Response: { "status": "success", "data": { "risk_score": 0.42,
                    "health_condition": "mild_risk", "risk_level": "Moderate", ... } }
        """
        start = time.time()
        try:
            # Backend expects flat features directly (not nested under "features")
            with httpx.Client(timeout=30) as client:
                resp = client.post(
                    f"{self.base_url}{self.predict_endpoint}",
                    json=features,          # send flat dict directly
                    headers=self.headers,
                )
                resp.raise_for_status()
                data = resp.json()
                inference_ms = (time.time() - start) * 1000

                if data.get("status") != "success":
                    return {
                        "success": False,
                        "error": data.get("message", "Backend returned non-success status"),
                        "inference_time_ms": inference_ms,
                    }

                payload = data["data"]
                return {
                    "success": True,
                    # Normalised prediction value (risk score 0–1 range)
                    "prediction": payload["risk_score"],
                    # Extra rich fields from the dog backend
                    "health_condition": payload.get("health_condition"),
                    "risk_level": payload.get("risk_level"),
                    "recommendation": payload.get("recommendation"),
                    "environmental_analysis": payload.get("environmental_analysis", {}),
                    "raw_response": data,
                    "inference_time_ms": inference_ms,
                }
        except httpx.HTTPStatusError as e:
            console.print(f"[red]Backend HTTP error {e.response.status_code}: {e.response.text}[/red]")
            return {"success": False, "error": str(e), "inference_time_ms": 0}
        except httpx.RequestError as e:
            console.print(f"[red]Backend connection error: {e}[/red]")
            return {"success": False, "error": str(e), "inference_time_ms": 0}

    def get_model_info(self) -> Dict[str, Any]:
        """GET /model-info"""
        try:
            with httpx.Client(timeout=10) as client:
                resp = client.get(
                    f"{self.base_url}{self.model_info_endpoint}",
                    headers=self.headers,
                )
                resp.raise_for_status()
                data = resp.json()
                cfg = data.get("configuration", {})
                perf = cfg.get("performance", {})
                return {
                    "model_version": f"catboost_dog_v{data.get('version', '3.0.0')}",
                    "model_type": data.get("model_type", "CatBoost Regressor"),
                    "base_features": data.get("feature_engineering", {}).get("base_features", []),
                    "final_features": data.get("feature_engineering", {}).get("final_features", []),
                    "val_r2": perf.get("val_r2"),
                    "val_rmse": perf.get("val_rmse"),
                    "thresholds": data.get("thresholds", {}),
                    "raw": data,
                }
        except Exception as e:
            console.print(f"[yellow]Could not fetch model info: {e}[/yellow]")
            return {
                "model_version": "catboost_dog_v3.0.0",
                "model_type": "CatBoost Regressor",
            }

    def get_features(self) -> Dict[str, Any]:
        """GET /features — returns required base features + example"""
        try:
            with httpx.Client(timeout=10) as client:
                resp = client.get(f"{self.base_url}/features", headers=self.headers)
                resp.raise_for_status()
                return resp.json()
        except Exception as e:
            console.print(f"[yellow]Could not fetch features: {e}[/yellow]")
            return {}

    def health_check(self) -> bool:
        """GET /health"""
        try:
            with httpx.Client(timeout=5) as client:
                resp = client.get(f"{self.base_url}/health", headers=self.headers)
                return resp.status_code == 200
        except Exception:
            return False