"""
Prediction Proxy — Dog Health Monitoring
Intercepts all predictions for monitoring before/after hitting the backend.
"""
import uuid
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
        """
        Full monitored prediction flow:
        1. Validate data quality
        2. Call dog health backend
        3. Log prediction + extra dog-specific fields
        4. Fire callbacks
        """
        # 1. Data quality
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
            return {
                "error": result.get("error", "Backend error"),
                "prediction": None,
                "prediction_id": None,
            }

        # 3. Log — store risk_score as prediction_value
        prediction_id = self.logger.log_prediction(
            features=features,
            prediction_value=result["prediction"],      # risk_score
            inference_time_ms=result["inference_time_ms"],
            metadata={
                **(metadata or {}),
                "health_condition": result.get("health_condition"),
                "risk_level": result.get("risk_level"),
                "recommendation": result.get("recommendation"),
                "environmental_analysis": result.get("environmental_analysis", {}),
            },
            quality_flags=quality_report,
        )

        response = {
            "prediction_id": prediction_id,
            # Core numeric score
            "risk_score": result["prediction"],
            # Human-readable dog health fields
            "health_condition": result.get("health_condition"),
            "risk_level": result.get("risk_level"),
            "recommendation": result.get("recommendation"),
            "environmental_analysis": result.get("environmental_analysis", {}),
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
                            actual_risk_score: float) -> Dict[str, Any]:
        """
        Log the true risk score for a past prediction.
        actual_risk_score: the real observed risk (0.0 – 1.0 or raw score).
        """
        return self.logger.log_ground_truth(prediction_id, actual_risk_score)