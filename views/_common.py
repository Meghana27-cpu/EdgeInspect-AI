"""Helpers shared by inspection pages."""
import streamlit as st
import pandas as pd
import config
from modules import reporting, ui


def detections_table(dets) -> pd.DataFrame:
    return pd.DataFrame([{"Defect": config.PRETTY.get(d.label, d.label), "Confidence": round(d.conf, 3),
                          "Severity (by size)": d.severity, "Size share": f"{d.area_ratio * 100:.2f}%",
                          "Box (x1,y1,x2,y2)": ", ".join(str(int(v)) for v in d.box)} for d in dets])


def render_result(r: dict, image=None, title: str | None = None):
    res = r["result"]
    if title:
        st.markdown(f"#### {title}")
    c1, c2 = st.columns(2)
    with c1:
        st.caption("Original")
        ui.show_img(image if image is not None else r["original_path"])
    with c2:
        st.caption("Annotated")
        ui.show_img(res.annotated)
    a, b, c, d = st.columns([1.3, 1, 1, 1])
    with a:
        ui.verdict(r["status"])
    b.markdown(ui.kpi("Defects", len(res.detections), f"threshold {r['conf_threshold']:.2f}", "red" if res.detections else "green"), unsafe_allow_html=True)
    c.markdown(ui.kpi("Model inference", f"{res.inference_ms:.1f} ms", "forward pass only", "cyan"), unsafe_allow_html=True)
    d.markdown(ui.kpi("Total latency", f"{res.total_ms:.1f} ms", f"≈ {1000 / res.total_ms:.0f} FPS equivalent" if res.total_ms else "", "blue"), unsafe_allow_html=True)
    if res.detections:
        ui.show_df(detections_table(res.detections))
        st.caption("Confidence = how certain the model is. Severity = defect size relative to the image. They are independent measures.")
    else:
        st.success("No defects above the configured threshold.")


def pdf_button(ids: list[int], key: str, title: str, label="Generate PDF report"):
    """Two-step because Streamlit buttons rerun the script: build, then offer download."""
    k = f"pdf_{key}"
    if st.button(label, key=f"btn_{k}"):
        with st.spinner("Building PDF..."):
            try:
                p = reporting.build_pdf(ids, title=title)
                st.session_state[k] = (p.name, p.read_bytes())
            except Exception as e:
                st.error(f"PDF generation failed: {e}")
    if k in st.session_state:
        name, data = st.session_state[k]
        st.download_button("Download PDF", data, file_name=name, mime="application/pdf", key=f"dl_{k}")
