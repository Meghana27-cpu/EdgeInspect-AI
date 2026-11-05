"""EdgeInspect AI - entry point.   Run:  streamlit run app.py"""
import importlib
import traceback
import streamlit as st

st.set_page_config(page_title="EdgeInspect AI", page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")

import config
from modules import services, ui

NAV = [
    ("Overview Dashboard", "dashboard"), ("Product Registration", "product_registration"),
    ("Image Inspection", "image_inspection"), ("Live Inspection", "live_inspection"),
    ("Multi-View Inspection", "multi_view"), ("Inspection History", "inspection_history"),
    ("Analytics & Reports", "analytics"), ("Model Performance", "model_performance"),
    ("Edge Optimization", "edge_optimization"), ("IoT Monitoring", "iot_monitoring"),
    ("System Settings", "settings"),
]

services.init()
ui.load_css()
with st.sidebar:
    st.markdown("<div class='brand'><div class='logo'>EI</div><div><div class='t1'>EdgeInspect AI</div>"
                "<div class='t2'>Industrial QC</div></div></div>", unsafe_allow_html=True)
    choice = st.radio("Navigation", [n for n, _ in NAV], label_visibility="collapsed", key="nav")
    st.markdown("---")
    det = services.detector()
    st.markdown(ui.pill("Model loaded" if det.loaded else "Model not loaded", "green" if det.loaded else "orange"), unsafe_allow_html=True)
    st.caption(f"{det.backend if det.loaded else 'no backend'}  ·  threshold {config.effective_conf(services.settings()):.2f}")

module = dict(NAV)[choice]
try:
    importlib.import_module(f"views.{module}").render()
except Exception as e:                      # a failing page must never take down the app
    st.error(f"This page hit an error: {type(e).__name__}: {e}")
    with st.expander("Technical details"):
        st.code(traceback.format_exc())
