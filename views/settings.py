import re
import socket
import streamlit as st
import config
from modules import edge_runtime as er, services, ui


def render():
    s = services.settings()
    ui.header("System Settings", "Configuration is validated before it is saved")
    with st.expander("Upload model weights (.pt / .onnx)"):
        up = st.file_uploader("Weights", type=["pt", "onnx"])
        if up and st.button("Save to models/"):
            name = re.sub(r"[^A-Za-z0-9_.\-]", "_", up.name)
            (config.MODELS_DIR / name).write_bytes(up.getvalue())
            st.success(f"Saved models/{name}. Select it below as the active model.")
    files = [f"models/{f.name}" for f in er.list_model_files()]
    with st.form("settings"):
        st.markdown("##### Model & inference")
        c = st.columns(3)
        mp = c[0].selectbox("Active model", files or [s["model_path"]], index=files.index(s["model_path"]) if s["model_path"] in files else 0) if files else c[0].text_input("Model path", s["model_path"])
        conf = c[1].slider("Confidence threshold", 0.01, 0.99, float(s["conf_threshold"]), 0.01)
        iou = c[2].slider("NMS IoU (PyTorch / legacy ONNX)", 0.05, 0.95, float(s["iou_threshold"]), 0.05)
        c = st.columns(4)
        imgsz = c[0].selectbox("Image size", config.CHOICES["imgsz"], index=config.CHOICES["imgsz"].index(s["imgsz"]) if s["imgsz"] in config.CHOICES["imgsz"] else 5)
        mode = c[1].selectbox("Inspection mode", config.CHOICES["inspection_mode"], index=config.CHOICES["inspection_mode"].index(s["inspection_mode"]))
        rt = c[2].selectbox("Edge runtime", config.CHOICES["edge_runtime"], index=config.CHOICES["edge_runtime"].index(s["edge_runtime"]))
        th = c[3].number_input("ONNX threads (0 = auto)", 0, 64, int(s["ort_threads"]))
        st.caption("High recall caps the threshold at 0.15; High precision floors it at 0.50. Static-shape ONNX models override the image size.")
        st.markdown("##### Sources")
        c = st.columns(2)
        cam = c[0].text_input("Camera source (index or URL)", s["camera_source"])
        sim = c[1].text_input("Simulation image folder", s["sim_dataset_dir"])
        st.markdown("##### MQTT")
        c = st.columns(4)
        me = c[0].checkbox("Use MQTT if available", s["mqtt_enabled"])
        mh = c[1].text_input("Broker host", s["mqtt_host"])
        mpn = c[2].number_input("Port", 1, 65535, int(s["mqtt_port"]))
        mpre = c[3].text_input("Topic prefix", s["mqtt_prefix"])
        st.markdown("##### Alert thresholds")
        c = st.columns(5)
        b0 = c[0].number_input("Belt speed min (m/s)", 0.0, 100.0, float(s["belt_speed_min"]), 0.1)
        b1 = c[1].number_input("Belt speed max (m/s)", 0.0, 100.0, float(s["belt_speed_max"]), 0.1)
        tm = c[2].number_input("Max temperature (°C)", 0.0, 500.0, float(s["temp_max"]), 1.0)
        vm = c[3].number_input("Max vibration (mm/s)", 0.0, 100.0, float(s["vibration_max"]), 0.5)
        c = st.columns(2)
        sm = c[0].number_input("Severity Medium from (% of image)", 0.01, 100.0, float(s["sev_medium_pct"]), 0.5)
        sh = c[1].number_input("Severity High from (% of image)", 0.01, 100.0, float(s["sev_high_pct"]), 0.5)
        ok = st.form_submit_button("Validate & save", type="primary")
    if ok:
        new, errs = config.validate_settings({"model_path": mp, "conf_threshold": conf, "iou_threshold": iou, "imgsz": imgsz, "inspection_mode": mode,
                                              "edge_runtime": rt, "ort_threads": th, "camera_source": cam, "sim_dataset_dir": sim, "mqtt_enabled": me,
                                              "mqtt_host": mh, "mqtt_port": mpn, "mqtt_prefix": mpre, "belt_speed_min": b0, "belt_speed_max": b1,
                                              "temp_max": tm, "vibration_max": vm, "sev_medium_pct": sm, "sev_high_pct": sh})
        if errs:
            for e in errs:
                st.error(e)
        else:
            services.apply_settings(new)
            if services.hub().running and (new["mqtt_host"], new["mqtt_port"], new["mqtt_enabled"]) != (s["mqtt_host"], s["mqtt_port"], s["mqtt_enabled"]):
                services.hub().stop()
                st.info("MQTT settings changed - the sensor hub was stopped; restart it on the IoT page.")
            st.success("Settings saved.")
            st.rerun()
    c1, c2, _ = st.columns([1, 1, 3])
    if c1.button("Test MQTT broker"):
        try:
            socket.create_connection((s["mqtt_host"], int(s["mqtt_port"])), timeout=2).close()
            st.success("Broker reachable.")
        except OSError as e:
            st.warning(f"Broker not reachable ({e}). The app will use local simulation.")
    if c2.button("Reset to defaults"):
        services.apply_settings(dict(config.DEFAULT_SETTINGS)); st.rerun()
    det = services.detector()
    st.markdown("##### Active model status")
    ui.status_rows([("Model file", s["model_path"], "blue"), ("State", "Loaded" if det.loaded else "Not loaded", "green" if det.loaded else "red"),
                    ("Backend", det.backend, "blue"), ("Class names", ", ".join(det.names), "gray")])
    if not det.loaded:
        ui.note(det.error, warn=True)
