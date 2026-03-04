"""
ML Monitoring Pipeline — Entry Point
After the pipeline starts, the user is prompted to choose a running mode:

  Mode 1 — Auto-Simulation   : Pipeline generates synthetic predictions on a timer
  Mode 2 — Continuous Feed   : UI-driven loop that fires predictions at a set interval
  Mode 3 — Production / API  : Pipeline listens passively; your app calls POST /predict
"""

import os
import signal
import sys
import time
import threading

import typer
import uvicorn
import yaml
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich import box

load_dotenv()
console = Console()
app_cli = typer.Typer(add_completion=False)


# ──────────────────────────────────────────────────────────────────────────────
# Config loader
# ──────────────────────────────────────────────────────────────────────────────

def load_config(config_path: str = "config/config.yaml") -> dict:
    cfg: dict = {}
    if os.path.exists(config_path):
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}
    cfg["ml_backend_url"] = os.getenv("ML_BACKEND_URL", cfg.get("ml_backend_url", "http://localhost:8000"))
    cfg["ml_api_key"]     = os.getenv("ML_API_KEY",     cfg.get("ml_api_key"))
    cfg["database_url"]   = os.getenv("DATABASE_URL",   cfg.get("database_url", "monitoring.db"))
    cfg["alert_channels"] = cfg.get("alert_channels", ["console"])
    return cfg


# ──────────────────────────────────────────────────────────────────────────────
# Mode selector — shown after pipeline starts
# ──────────────────────────────────────────────────────────────────────────────

def prompt_mode_selection(pipeline, host: str, port: int) -> None:
    """Interactive prompt asking the user which running mode to use."""

    console.print()
    console.rule("[bold cyan]Choose Running Mode[/bold cyan]")
    console.print()

    table = Table(box=box.ROUNDED, show_header=True, header_style="bold cyan",
                  border_style="dim", padding=(0, 2))
    table.add_column("Mode", style="bold yellow", width=6)
    table.add_column("Name", style="bold white", width=24)
    table.add_column("Description", style="dim")

    table.add_row(
        "1",
        "Auto-Simulation",
        "Pipeline generates synthetic dog predictions on a timer.\n"
        "  Great for testing monitoring, drift detection, and alerts\n"
        "  without any external input.",
    )
    table.add_row(
        "2",
        "Continuous Frontend Feed",
        "Use the dashboard UI to start/stop a prediction loop.\n"
        "  You control the interval and volume from the browser.\n"
        "  Visit http://localhost:" + str(port) + "/dashboard after selecting.",
    )
    table.add_row(
        "3",
        "Production / API Mode",
        "Pipeline listens passively. Your app calls POST /predict.\n"
        "  This is the intended real-world usage.\n"
        "  Swagger: http://localhost:" + str(port) + "/docs",
    )

    console.print(table)
    console.print()

    choice = Prompt.ask(
        "[bold cyan]Select mode[/bold cyan]",
        choices=["1", "2", "3"],
        default="3",
    )

    if choice == "1":
        _configure_auto_simulation(pipeline)
    elif choice == "2":
        _configure_continuous_feed(pipeline, host, port)
    else:
        _configure_production_mode(host, port)


def _configure_auto_simulation(pipeline) -> None:
    """Mode 1 — ask interval + count, start simulation inside the pipeline."""
    console.print()
    console.print(Panel(
        "[bold green]Mode 1 — Auto-Simulation[/bold green]\n"
        "The pipeline will automatically generate synthetic dog predictions\n"
        "and send them to your backend on a timer.",
        border_style="green",
    ))

    interval = float(Prompt.ask(
        "  Prediction interval [bold](seconds)[/bold]",
        default="5",
    ))
    max_preds = int(Prompt.ask(
        "  Max predictions [bold](0 = unlimited)[/bold]",
        default="0",
    ))
    auto_gt = Confirm.ask(
        "  Auto-submit simulated ground truth?",
        default=True,
    )

    import random

    def dog_feature_generator():
        breeds = [
            "Labrador Retriever", "German Shepherd", "Golden Retriever",
            "Bulldog", "Pug", "Beagle", "Poodle", "Rottweiler",
            "Siberian Husky", "Dachshund", "Yorkshire Terrier", "Boxer",
        ]
        return {
            "breed":               random.choice(breeds),
            "weight_kg":           round(random.uniform(4.0, 45.0), 1),
            "heart_rate_bpm":      round(random.uniform(55.0, 170.0), 1),
            "temperature_celsius": round(random.uniform(10.0, 40.0), 1),
            "humidity_pct":        round(random.uniform(20.0, 95.0), 1),
            "speed_kmh":           round(random.uniform(0.0, 20.0), 1),
        }

    def ground_truth_fn(result):
        pred = result.get("prediction", 0.5)
        return round(max(0.0, min(1.0, pred + random.uniform(-0.08, 0.08))), 4)

    pipeline.start_simulation(
        feature_generator=dog_feature_generator,
        interval_seconds=interval,
        max_predictions=max_preds,
        auto_ground_truth=auto_gt,
        ground_truth_fn=ground_truth_fn if auto_gt else None,
    )

    console.print()
    console.print(Panel(
        f"[bold green]▶ Auto-simulation running[/bold green]\n\n"
        f"  Interval  : [yellow]{interval}s[/yellow]\n"
        f"  Max preds : [yellow]{'unlimited' if max_preds == 0 else max_preds}[/yellow]\n"
        f"  Auto GT   : [yellow]{auto_gt}[/yellow]\n\n"
        "  Press [bold]Ctrl+C[/bold] to stop the pipeline.",
        border_style="green",
    ))


def _configure_continuous_feed(pipeline, host: str, port: int) -> None:
    """Mode 2 — tell user to use the dashboard UI to control the feed."""
    console.print()
    console.print(Panel(
        "[bold yellow]Mode 2 — Continuous Frontend Feed[/bold yellow]\n\n"
        "  The pipeline is ready. Control the prediction loop from the dashboard:\n\n"
        f"  [bold cyan]http://localhost:{port}/dashboard[/bold cyan]\n\n"
        "  From the dashboard you can:\n"
        "    • Set the prediction interval\n"
        "    • Set max predictions (or run unlimited)\n"
        "    • Enable auto ground-truth submission\n"
        "    • Start / stop the feed at any time\n"
        "    • Watch live charts update in real-time\n\n"
        "  Press [bold]Ctrl+C[/bold] to stop the pipeline.",
        border_style="yellow",
    ))


def _configure_production_mode(host: str, port: int) -> None:
    """Mode 3 — pipeline is passive, just print the API endpoints."""
    console.print()
    console.print(Panel(
        "[bold blue]Mode 3 — Production / API Mode[/bold blue]\n\n"
        "  The pipeline is listening. Send predictions from your application:\n\n"
        f"  [bold]POST[/bold]  http://localhost:{port}/predict\n"
        f"  [bold]POST[/bold]  http://localhost:{port}/ground_truth\n"
        f"  [bold]GET[/bold]   http://localhost:{port}/dashboard\n"
        f"  [bold]GET[/bold]   http://localhost:{port}/metrics/performance\n"
        f"  [bold]GET[/bold]   http://localhost:{port}/metrics/drift\n"
        f"  [bold]GET[/bold]   http://localhost:{port}/alerts\n\n"
        f"  Swagger UI: [bold cyan]http://localhost:{port}/docs[/bold cyan]\n\n"
        "  Press [bold]Ctrl+C[/bold] to stop the pipeline.",
        border_style="blue",
    ))


# ──────────────────────────────────────────────────────────────────────────────
# CLI commands
# ──────────────────────────────────────────────────────────────────────────────

@app_cli.command()
def start(
    backend_url:         str = typer.Option(None,           "--backend-url",         "-b",  help="ML backend URL"),
    api_key:             str = typer.Option(None,           "--api-key",             "-k",  help="API key for backend"),
    adapter:             str = typer.Option("flat",         "--adapter",             "-a",  help="flat|nested|wrapped|classification|list|dog_health"),
    prediction_field:    str = typer.Option("prediction",   "--prediction-field",           help="Response field for prediction value"),
    data_key:            str = typer.Option("data",         "--data-key",                   help="Outer key for nested adapters"),
    predict_endpoint:    str = typer.Option("/predict",     "--predict-endpoint",           help="Predict endpoint path"),
    model_info_endpoint: str = typer.Option("/model/info",  "--model-info-endpoint",        help="Model info endpoint path"),
    health_endpoint:     str = typer.Option("/health",      "--health-endpoint",            help="Health check endpoint path"),
    config:              str = typer.Option("config/config.yaml", "--config",        "-c",  help="Config file path"),
    host:                str = typer.Option("0.0.0.0",      "--host",                       help="Dashboard host"),
    port:                int = typer.Option(8080,           "--port",                       help="Dashboard port"),
    mode:                int = typer.Option(0,              "--mode",                "-m",  help="Skip prompt: 1=simulation, 2=frontend, 3=production"),
):
    """Start the ML Model Monitoring Pipeline."""

    from pipeline.orchestrator import MLMonitoringPipeline
    from dashboard.app import app as fastapi_app, set_pipeline

    cfg = load_config(config)
    if backend_url:  cfg["ml_backend_url"]   = backend_url
    if api_key:      cfg["ml_api_key"]        = api_key

    cfg["predict_endpoint"]    = predict_endpoint
    cfg["model_info_endpoint"] = model_info_endpoint
    cfg["health_endpoint"]     = health_endpoint
    cfg["adapter"] = {
        "type": adapter,
        "prediction_field": prediction_field,
        **({"data_key": data_key} if adapter == "nested" else {}),
    }

    # ── Banner ────────────────────────────────────────────────────────────────
    console.print()
    console.print(Panel(
        f"[bold cyan]ML Model Monitoring Pipeline[/bold cyan]\n\n"
        f"  Backend   : [yellow]{cfg['ml_backend_url']}[/yellow]\n"
        f"  Adapter   : [yellow]{adapter}[/yellow]\n"
        f"  Dashboard : [yellow]http://localhost:{port}[/yellow]\n"
        f"  Swagger   : [yellow]http://localhost:{port}/docs[/yellow]",
        border_style="cyan",
    ))

    # ── Init pipeline ─────────────────────────────────────────────────────────
    pipeline = MLMonitoringPipeline(cfg)
    pipeline.start()
    set_pipeline(pipeline)

    # ── Mode selection ────────────────────────────────────────────────────────
    if mode in (1, 2, 3):
        # Non-interactive: --mode flag was passed
        if mode == 1:
            _configure_auto_simulation(pipeline)
        elif mode == 2:
            _configure_continuous_feed(pipeline, host, port)
        else:
            _configure_production_mode(host, port)
    else:
        # Interactive prompt
        prompt_mode_selection(pipeline, host, port)

    # ── Graceful shutdown ─────────────────────────────────────────────────────
    def shutdown(sig, frame):
        console.print("\n[yellow]Shutting down pipeline…[/yellow]")
        try:
            pipeline.stop_simulation()
        except Exception:
            pass
        pipeline.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT,  shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # ── Start dashboard server ────────────────────────────────────────────────
    uvicorn.run(fastapi_app, host=host, port=port, log_level="warning")


@app_cli.command()
def demo():
    """Quick offline demo — no backend needed. Runs Mode 1 simulation with stub."""
    console.print(Panel(
        "[bold green]Demo Mode[/bold green] — running 10 synthetic predictions offline",
        border_style="green",
    ))
    import random

    cfg = load_config()
    cfg["ml_backend_url"] = "http://localhost:9999"   # won't be called
    cfg["adapter"] = {"type": "flat"}
    cfg["baseline_stats"] = {
        "feature_1": {"type": "numerical", "mean": 10.0, "std": 2.0,
                      "min": 0.0, "max": 20.0, "missing_rate": 0.0},
        "feature_2": {"type": "numerical", "mean": 0.5,  "std": 0.1,
                      "min": 0.0, "max": 1.0,  "missing_rate": 0.0},
        "category":  {"type": "categorical", "categories": ["A", "B", "C"],
                      "category_distribution": {"A": 0.5, "B": 0.3, "C": 0.2}},
    }

    from pipeline.orchestrator import MLMonitoringPipeline
    pipeline = MLMonitoringPipeline(cfg)

    ids = []
    for i in range(10):
        result = pipeline.predict({
            "feature_1": random.gauss(10, 2),
            "feature_2": random.uniform(0, 1),
            "category":  random.choice(["A", "B", "C"]),
        })
        if result.get("prediction_id"):
            ids.append(result["prediction_id"])
        console.print(f"  [{i+1:02d}] {result}")

    for pid in ids[:5]:
        pipeline.submit_ground_truth(pid, random.gauss(0.3, 0.1))

    pipeline.run_daily_evaluation()

    import json
    console.print_json(json.dumps(pipeline.get_dashboard_data(), indent=2, default=str))


if __name__ == "__main__":
    app_cli()