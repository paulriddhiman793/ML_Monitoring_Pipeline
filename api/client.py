"""
ML Backend Client — generic, adapter-driven
"""
import time
import httpx
from typing import Any, Dict, Optional
from rich.console import Console
from api.adapters import BaseAdapter, FlatPredictionAdapter

console = Console()


class MLBackendClient:

    def __init__(self, base_url: str, adapter: BaseAdapter = None,
                 api_key: Optional[str] = None,
                 predict_endpoint: str = "/predict",
                 model_info_endpoint: str = "/model/info",
                 health_endpoint: str = "/health"):
        self.base_url = base_url.rstrip("/")
        self.adapter = adapter or FlatPredictionAdapter()
        self.predict_endpoint = predict_endpoint
        self.model_info_endpoint = model_info_endpoint
        self.health_endpoint = health_endpoint
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"

    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        start = time.time()
        try:
            body = self.adapter.build_request_body(features)
            with httpx.Client(timeout=30) as client:
                resp = client.post(
                    f"{self.base_url}{self.predict_endpoint}",
                    json=body,
                    headers=self.headers,
                )
                resp.raise_for_status()
                data = resp.json()
                inference_ms = (time.time() - start) * 1000

            prediction = self.adapter.extract_prediction(data)
            if prediction is None:
                return {
                    "success": False,
                    "error": f"Could not extract prediction from response: {data}",
                    "raw_response": data,
                    "inference_time_ms": inference_ms,
                }

            return {
                "success": True,
                "prediction": prediction,
                "extra": self.adapter.extract_extra(data),
                "raw_response": data,
                "inference_time_ms": inference_ms,
            }
        except httpx.HTTPStatusError as e:
            console.print(f"[red]Backend HTTP {e.response.status_code}: {e.response.text[:200]}[/red]")
            return {"success": False, "error": str(e), "inference_time_ms": 0}
        except httpx.RequestError as e:
            console.print(f"[red]Backend connection error: {e}[/red]")
            return {"success": False, "error": str(e), "inference_time_ms": 0}

    def get_model_info(self) -> Dict[str, Any]:
        try:
            with httpx.Client(timeout=10) as client:
                resp = client.get(
                    f"{self.base_url}{self.model_info_endpoint}",
                    headers=self.headers,
                )
                resp.raise_for_status()
                raw = resp.json()
                return {
                    "model_version": self.adapter.get_model_version(raw),
                    "baseline_metrics": self.adapter.get_baseline_metrics(raw),
                    "raw": raw,
                }
        except Exception as e:
            console.print(f"[yellow]Could not fetch model info: {e}[/yellow]")
            return {"model_version": "v1", "baseline_metrics": {}, "raw": {}}

    def health_check(self) -> bool:
        try:
            with httpx.Client(timeout=5) as client:
                resp = client.get(
                    f"{self.base_url}{self.health_endpoint}",
                    headers=self.headers,
                )
                return resp.status_code == 200
        except Exception:
            return False