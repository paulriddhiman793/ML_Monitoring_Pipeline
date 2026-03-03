"""
Prediction Proxy — fully generic
"""
from datetime import datetime
from typing import Any, Callable, Dict, Optional

from api.client import MLBackendClient
from core.logger import MLMonitoringLogger
from core.data_quality import DataQualityMonitor


class PredictionProxy:

    def __init__(self, backend_client: MLBackendClient,
                 logger: MLMonitoringLogger,
                 quality_monitor: DataQualityMonitor):
        self.backend = backend_client
        self.logger = logger
        self.quality_monitor = quality_monitor
        self._callbacks: list[Callable] = []

    def add_prediction_callback(self, fn: Callable):
        self._callbacks.append(fn)

    def predict(self, features: Dict[str, Any],
                metadata: Optional[Dict] = None) -> Dict[str, Any]:
        # 1. Quality check
        quality_report = self.quality_monitor.validate_input(features)
        self.logger.log_quality(quality_report)

        if not quality_report["valid"]:
            return {
                "error": "Data quality issues — prediction blocked",
                "quality_report": quality_report,
                "prediction": None,
                "prediction_id": None,
            }

        # 2. Backend call
        result = self.backend.predict(features)
        if not result["success"]:
            return {"error": result.get("error"), "prediction": None, "prediction_id": None}

        # 3. Log — extra fields go into metadata (adapter-provided, domain-agnostic)
        prediction_id = self.logger.log_prediction(
            features=features,
            prediction_value=result["prediction"],
            inference_time_ms=result["inference_time_ms"],
            metadata={**(metadata or {}), **result.get("extra", {})},
            quality_flags=quality_report,
        )

        response = {
            "prediction_id": prediction_id,
            "prediction": result["prediction"],
            # Extra fields pass through transparently — whatever the adapter extracted
            **result.get("extra", {}),
            "inference_time_ms": result["inference_time_ms"],
            "quality_warnings": quality_report.get("warnings", []),
        }

        for cb in self._callbacks:
            try:
                cb(response)
            except Exception:
                pass

        return response

    def submit_ground_truth(self, prediction_id: str,
                            actual_value: float) -> Dict[str, Any]:
        return self.logger.log_ground_truth(prediction_id, actual_value)