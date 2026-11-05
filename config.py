"""Central configuration, defaults and settings validation."""
from __future__ import annotations
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
DB_DIR = BASE_DIR / "database"
UPLOAD_DIR = BASE_DIR / "uploads"
RESULTS_DIR = BASE_DIR / "results"
REPORTS_DIR = BASE_DIR / "reports"
DATASETS_DIR = BASE_DIR / "datasets"
DB_PATH = DB_DIR / "edgeinspect.db"
SETTINGS_PATH = DB_DIR / "settings.json"

PROJECT_NAME = "Real-Time Edge-AI Visual Inspection System for Automated Industrial Defect Detection"
APP_NAME = "EdgeInspect AI"

# NEU-DET steel surface classes (order must match the YOLO dataset yaml)
CLASS_NAMES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
PRETTY = {
    "crazing": "Crazing", "inclusion": "Inclusion", "patches": "Patches",
    "pitted_surface": "Pitted Surface", "rolled-in_scale": "Rolled-in Scale", "scratches": "Scratches",
}
PRODUCT_CATEGORIES = ["Hot-rolled steel strip", "Steel plate", "Steel coil", "Fabricated part", "Other"]
VIEW_NAMES = ["Front", "Back", "Left", "Right", "Top", "Bottom", "Other"]

TARGET_LATENCY_MS = 15.0
TARGET_RECALL = 0.98

# Edge profiles. SIMULATED profiles only cap CPU threads on the host machine;
# they do NOT reproduce the real clock speed / memory bandwidth of the device.
EDGE_PROFILES = {
    "Host machine (actual)": {"threads": 0, "simulated": False},
    "Raspberry Pi 4 (simulated: 4 threads)": {"threads": 4, "simulated": True},
    "Raspberry Pi 5 (simulated: 4 threads)": {"threads": 4, "simulated": True},
    "Jetson Orin Nano (simulated, CPU-only: 6 threads)": {"threads": 6, "simulated": True},
}

DEFAULT_SETTINGS = {
    "model_path": "models/best.onnx",
    "conf_threshold": 0.25,
    "iou_threshold": 0.45,
    "imgsz": 640,
    "inspection_mode": "Standard",      # Standard | High recall | High precision
    "edge_runtime": "auto",             # auto | onnxruntime-cpu | onnxruntime-gpu | pytorch
    "ort_threads": 0,
    "camera_source": "0",
    "sim_dataset_dir": "datasets/neu-det/images/val",
    "mqtt_enabled": True,
    "mqtt_host": "localhost",
    "mqtt_port": 1883,
    "mqtt_prefix": "edgeinspect",
    "belt_speed_min": 0.5, "belt_speed_max": 2.5,
    "temp_max": 70.0, "vibration_max": 6.0,
    "sev_medium_pct": 2.0, "sev_high_pct": 8.0,
}
CHOICES = {
    "inspection_mode": ["Standard", "High recall", "High precision"],
    "edge_runtime": ["auto", "onnxruntime-cpu", "onnxruntime-gpu", "pytorch"],
    "imgsz": [160, 224, 320, 416, 512, 640, 800, 1024],
}


def ensure_dirs() -> None:
    for d in (MODELS_DIR, DB_DIR, UPLOAD_DIR, RESULTS_DIR, REPORTS_DIR, DATASETS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else BASE_DIR / p


def validate_settings(raw: dict) -> tuple[dict, list[str]]:
    """Return (clean_settings, errors). Unknown keys are dropped, bad values reported."""
    s = dict(DEFAULT_SETTINGS)
    errors: list[str] = []

    def num(key, lo, hi, cast=float):
        try:
            v = cast(raw.get(key, s[key]))
            if not lo <= v <= hi:
                raise ValueError
            s[key] = v
        except (TypeError, ValueError):
            errors.append(f"{key} must be a number between {lo} and {hi}")

    num("conf_threshold", 0.01, 0.99); num("iou_threshold", 0.05, 0.95)
    num("ort_threads", 0, 64, int); num("mqtt_port", 1, 65535, int)
    num("belt_speed_min", 0, 100); num("belt_speed_max", 0, 100)
    num("temp_max", 0, 500); num("vibration_max", 0, 100)
    num("sev_medium_pct", 0.01, 100); num("sev_high_pct", 0.01, 100)
    try:
        v = int(raw.get("imgsz", s["imgsz"]))
        if v < 32 or v > 2048 or v % 32:
            raise ValueError
        s["imgsz"] = v
    except (TypeError, ValueError):
        errors.append("imgsz must be a multiple of 32 between 32 and 2048")
    for k, opts in CHOICES.items():
        if k == "imgsz":
            continue
        v = raw.get(k, s[k])
        if v in opts:
            s[k] = v
        else:
            errors.append(f"{k} must be one of {opts}")
    mp = str(raw.get("model_path", s["model_path"])).strip()
    if not mp.lower().endswith((".pt", ".onnx")):
        errors.append("model_path must end with .pt or .onnx")
    else:
        s["model_path"] = mp
    for k in ("camera_source", "sim_dataset_dir", "mqtt_host", "mqtt_prefix"):
        v = str(raw.get(k, s[k])).strip()
        if v:
            s[k] = v
        else:
            errors.append(f"{k} must not be empty")
    s["mqtt_enabled"] = bool(raw.get("mqtt_enabled", s["mqtt_enabled"]))
    if s["belt_speed_min"] >= s["belt_speed_max"]:
        errors.append("belt_speed_min must be lower than belt_speed_max")
    if s["sev_medium_pct"] >= s["sev_high_pct"]:
        errors.append("sev_medium_pct must be lower than sev_high_pct")
    return s, errors


def load_settings() -> dict:
    raw = {}
    try:
        raw = json.loads(SETTINGS_PATH.read_text())
    except (OSError, ValueError):
        pass
    clean, _ = validate_settings(raw)
    return clean


def save_settings(s: dict) -> None:
    DB_DIR.mkdir(parents=True, exist_ok=True)
    SETTINGS_PATH.write_text(json.dumps(s, indent=2))


def effective_conf(s: dict) -> float:
    """Inspection mode adjusts the configured threshold."""
    t = float(s["conf_threshold"])
    if s["inspection_mode"] == "High recall":
        return min(t, 0.15)
    if s["inspection_mode"] == "High precision":
        return max(t, 0.50)
    return t
