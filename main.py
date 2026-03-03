import os
import signal
import sys
import time

import typer
import uvicorn
import yaml
from dotenv import load_dotenv
from rich.console import Console

load_dotenv()
console = Console()
app_cli = typer.Typer()


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f) or {}
    cfg["ml_backend_url"] = os.getenv("ML_BACKEND_URL", "http://localhost:8000")
    cfg["ml_api_key"]     = os.getenv("ML_API_KEY")
    cfg["database_url"]   = os.getenv("DATABASE_URL", "monitoring.db")
    cfg["alert_channels"] = ["console"]
    return cfg


@app_cli.command()
def start(
    backend_url:          str = typer.Option(None,          "--backend-url",          "-b"),
    api_key:              str = typer.Option(None,          "--api-key",              "-k"),
    adapter:              str = typer.Option("flat",        "--adapter",              "-a",
        help="flat | nested | wrapped | classification | list | dog_health"),
    prediction_field:     str = typer.Option("prediction",  "--prediction-field"),
    data_key:             str = typer.Option("data",        "--data-key"),
    predict_endpoint:     str = typer.Option("/predict",    "--predict-endpoint"),
    model_info_endpoint:  str = typer.Option("/model/info", "--model-info-endpoint"),
    health_endpoint:      str = typer.Option("/health",     "--health-endpoint"),
    config:               str = typer.Option("config/config.yaml", "--config", "-c"),
    host:                 str = typer.Option("0.0.0.0",     "--host"),
    port:                 int = typer.Option(8080,          "--port"),
):
    """Start the ML Model Monitoring Pipeline."""
    from pipeline.orchestrator import MLMonitoringPipeline
    from dashboard.app import app as fastapi_app, set_pipeline

    cfg = load_config(config)

    if backend_url:        cfg["ml_backend_url"]   = backend_url
    if api_key:            cfg["ml_api_key"]        = api_key
    cfg["predict_endpoint"]    = predict_endpoint
    cfg["model_info_endpoint"] = model_info_endpoint
    cfg["health_endpoint"]     = health_endpoint

    # Build adapter config from CLI
    cfg["adapter"] = {"type": adapter, "prediction_field": prediction_field}
    if adapter == "nested":
        cfg["adapter"]["data_key"] = data_key

    console.print(f"\n[bold cyan]ML Monitoring Pipeline[/bold cyan]")
    console.print(f"Backend  : [yellow]{cfg['ml_backend_url']}[/yellow]")
    console.print(f"Adapter  : [yellow]{adapter}[/yellow]")
    console.print(f"Dashboard: [yellow]http://{host}:{port}/docs[/yellow]\n")

    pipeline = MLMonitoringPipeline(cfg)
    pipeline.start()
    set_pipeline(pipeline)

    def shutdown(sig, frame):
        console.print("\n[yellow]Shutting down…[/yellow]")
        pipeline.stop()
        sys.exit(0)
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    uvicorn.run(fastapi_app, host=host, port=port, log_level="warning")


@app_cli.command()
def demo(
    backend_url: str = typer.Option("http://localhost:8000", "--backend-url"),
    adapter:     str = typer.Option("flat",                  "--adapter"),
):
    """Quick demo with synthetic predictions — no real backend needed for flat adapter."""
    import random
    from pipeline.orchestrator import MLMonitoringPipeline

    cfg = load_config()
    cfg["ml_backend_url"] = backend_url
    cfg["adapter"] = {"type": adapter}
    cfg["baseline_stats"] = {
        "feature_1": {"type": "numerical", "mean": 10.0, "std": 2.0,
                      "min": 0.0, "max": 20.0, "missing_rate": 0.0},
        "feature_2": {"type": "numerical", "mean": 0.5,  "std": 0.1,
                      "min": 0.0, "max": 1.0,  "missing_rate": 0.0},
        "category":  {"type": "categorical", "categories": ["A", "B", "C"],
                      "category_distribution": {"A": 0.5, "B": 0.3, "C": 0.2}},
    }

    pipeline = MLMonitoringPipeline(cfg)
    console.print("[bold green]Sending 10 synthetic predictions…[/bold green]")
    ids = []
    for i in range(10):
        r = pipeline.predict({
            "feature_1": random.gauss(10, 2),
            "feature_2": random.uniform(0, 1),
            "category":  random.choice(["A", "B", "C"]),
        })
        if r.get("prediction_id"):
            ids.append(r["prediction_id"])
        console.print(f"  [{i+1}] {r}")

    for pid in ids[:5]:
        pipeline.submit_ground_truth(pid, random.gauss(0.3, 0.1))

    pipeline.run_daily_evaluation()
    import json
    console.print_json(json.dumps(pipeline.get_dashboard_data(), indent=2, default=str))


if __name__ == "__main__":
    app_cli()