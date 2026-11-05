import tempfile
import time
from collections import deque
from pathlib import Path
import cv2
import numpy as np
import streamlit as st
import config
from modules import database as db, pipeline, services, ui
from modules.preprocessing import list_images

SAVE_EVERY_S = 2.0


def _frames(mode, src, fps, loop):
    """Yield (frame, label). Dataset mode is a *simulated* conveyor built from still images."""
    if mode == "Dataset conveyor simulation":
        imgs = list_images(src)
        if not imgs:
            raise RuntimeError(f"No images found in '{src}'. Set the folder (e.g. datasets/neu-det/images/val).")
        while True:
            for p in imgs:
                im = cv2.imread(str(p))
                if im is not None:
                    yield im, f"SIMULATED conveyor · {p.name}"
            if not loop:
                return
    cap = cv2.VideoCapture(int(src) if str(src).isdigit() else str(src))
    if not cap.isOpened():
        st.session_state["camera_status"] = "Disconnected"
        raise RuntimeError(f"Cannot open video source '{src}'.")
    if mode.startswith("Webcam"):
        st.session_state["camera_status"] = "Connected"
    try:
        while True:
            ok, fr = cap.read()
            if not ok:
                if loop and not mode.startswith("Webcam"):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0); continue
                return
            yield fr, "LIVE camera" if mode.startswith("Webcam") else "Video file"
    finally:
        cap.release()


def render():
    det, s = services.detector(), services.settings()
    ui.header("Live Inspection", "Frame-by-frame inference on a camera, a video file, or a simulated conveyor",
              ui.pill("RUNNING", "green") if st.session_state.get("live_on") else ui.pill("STOPPED", "gray"))
    if services.need_model(det):
        return
    mode = st.radio("Input source", ["Webcam / camera", "Upload video file", "Dataset conveyor simulation"], horizontal=True)
    src, fps_target = None, 10
    c1, c2, c3 = st.columns(3)
    if mode == "Webcam / camera":
        src = c1.text_input("Camera index or stream URL", s["camera_source"])
        if c2.button("Probe camera"):
            cap = cv2.VideoCapture(int(src) if src.isdigit() else src)
            st.session_state["camera_status"] = "Connected" if cap.isOpened() else "Disconnected"
            cap.release()
        c3.markdown(ui.pill("Camera: " + st.session_state.get("camera_status", "Not checked"),
                            "green" if st.session_state.get("camera_status") == "Connected" else "orange"), unsafe_allow_html=True)
        ui.note("Browser-hosted deployments (cloud) cannot open a server-side webcam; use the video or simulation mode there.")
    elif mode == "Upload video file":
        up = c1.file_uploader("Video", type=["mp4", "avi", "mov", "mkv"])
        if up:
            tmp = Path(tempfile.gettempdir()) / f"edgeinspect_{up.name}"
            tmp.write_bytes(up.getvalue()); src = str(tmp)
    else:
        src = c1.text_input("Image folder", s["sim_dataset_dir"])
        fps_target = c2.slider("Conveyor speed (frames/s)", 1, 30, 10)
        ui.note("<b>Simulation:</b> still images from the dataset are fed one by one to mimic a conveyor. This tests the real-time pipeline, "
                "but it is not a physical camera feed.")
    loop = st.checkbox("Loop source", mode != "Upload video file")
    pids = db.list_products().product_id.tolist()
    pid = st.selectbox("Link detections to product (optional)", ["- none -"] + pids)
    save = st.checkbox("Save defect frames to database (max 1 per 2 s)", True)
    b1, b2, _ = st.columns([1, 1, 4])
    if b1.button("Start inspection", type="primary", disabled=src is None):
        st.session_state["live_on"] = True
    if b2.button("Stop"):
        st.session_state["live_on"] = False
    frame_ph, stats_ph, log_ph = st.empty(), st.empty(), st.empty()
    if not st.session_state.get("live_on"):
        ui.empty("Inspection stopped", "Choose a source and press Start inspection.")
        return
    conf = config.effective_conf(s)
    lat, log = deque(maxlen=120), deque(maxlen=6)
    n_frames = n_def = 0
    last_save, t_prev, fps_ema = 0.0, time.perf_counter(), 0.0
    try:
        for frame, label in _frames(mode, src, fps_target, loop):
            t0 = time.perf_counter()
            res = det.predict(frame, conf=conf, iou=s["iou_threshold"])
            lat.append(res.total_ms)
            n_frames += 1
            status = pipeline.decide(res.detections)
            if res.detections:
                n_def += len(res.detections)
                if time.time() - last_save > SAVE_EVERY_S:
                    last_save = time.time()
                    top = res.detections[0]
                    log.appendleft(f"{time.strftime('%H:%M:%S')}  ALERT  {config.PRETTY.get(top.label, top.label)} {top.conf:.2f}  ({len(res.detections)} defect(s))")
                    if save:
                        pipeline.persist_result(det, frame, res, conf, None if pid == "- none -" else pid, "live", hub=services.hub())
                    else:
                        services.hub().publish_event("alerts/defect", {"defects": len(res.detections)})
            now = time.perf_counter()
            inst = 1.0 / max(now - t_prev, 1e-6); t_prev = now
            fps_ema = inst if fps_ema == 0 else 0.9 * fps_ema + 0.1 * inst
            frame_ph.image(cv2.cvtColor(res.annotated, cv2.COLOR_BGR2RGB), caption=f"{label} · frame {n_frames}", width=720)
            tone = "red" if status == "FAIL" else "green"
            stats_ph.markdown(
                "<div style='display:grid;grid-template-columns:repeat(5,1fr);gap:.7rem'>"
                + ui.kpi("Result", status, f"{len(res.detections)} defect(s) in frame", tone)
                + ui.kpi("Display FPS", f"{fps_ema:.1f}", "full loop incl. drawing", "cyan")
                + ui.kpi("Latency", f"{res.total_ms:.1f} ms", f"p95 {np.percentile(lat, 95):.1f} ms", "green" if res.total_ms < config.TARGET_LATENCY_MS else "orange")
                + ui.kpi("Inference", f"{res.inference_ms:.1f} ms", "model forward pass", "blue")
                + ui.kpi("Totals", f"{n_def}", f"defects in {n_frames} frames", "orange") + "</div>", unsafe_allow_html=True)
            log_ph.code("\n".join(log) or "No alerts yet", language="text")
            if mode == "Dataset conveyor simulation":
                time.sleep(max(0.0, 1.0 / fps_target - (time.perf_counter() - t0)))
        st.session_state["live_on"] = False
        st.info("Source finished.")
    except RuntimeError as e:
        st.session_state["live_on"] = False
        st.error(str(e))
