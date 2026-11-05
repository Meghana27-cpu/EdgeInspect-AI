"""Streamlit-aware singletons: settings, detector, sensor hub."""
from __future__ import annotations
import streamlit as st
import config
from modules import database as db
from modules.detector import Detector
from modules.mqtt_handler import SensorHub


def settings() -> dict:
    if "settings" not in st.session_state:
        st.session_state["settings"] = config.load_settings()
    return st.session_state["settings"]


def apply_settings(new: dict):
    st.session_state["settings"] = new
    config.save_settings(new)


@st.cache_resource(show_spinner="Loading model...")
def _detector(path, imgsz, runtime, threads, mtime, med, high):
    return Detector(path, imgsz, runtime, threads, medium_pct=med, high_pct=high)


def detector(threads_override: int | None = None) -> Detector:
    s = settings()
    p = config.resolve(s["model_path"])
    mt = p.stat().st_mtime if p.exists() else 0
    return _detector(s["model_path"], s["imgsz"], s["edge_runtime"],
                     s["ort_threads"] if threads_override is None else threads_override, mt,
                     s["sev_medium_pct"], s["sev_high_pct"])


@st.cache_resource
def _hub():
    return SensorHub(settings())


def hub() -> SensorHub:
    h = _hub()
    h.s = settings()          # live-update thresholds
    return h


def init():
    config.ensure_dirs()
    db.init_db()


def need_model(det: Detector) -> bool:
    """Show a friendly message and return True if the page must stop."""
    if det.loaded:
        return False
    from modules import ui
    ui.note(f"<b>AI model not loaded.</b> {det.error}<br>Place trained weights (e.g. <code>best.pt</code> / <code>best.onnx</code>) in "
            "<code>models/</code> or upload them in <b>System Settings</b>, then check the model path.", warn=True)
    return True
