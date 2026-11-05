import streamlit as st
import config
from modules import database as db, pipeline, services, ui
from modules.preprocessing import decode_image
from views._common import pdf_button, render_result

ADHOC = "- none (ad-hoc inspection) -"


def render():
    det, s = services.detector(), services.settings()
    ui.header("Image Inspection", "Upload or capture an image and run AI defect detection",
              ui.pill(f"{det.backend}", "green") if det.loaded else ui.pill("Model not loaded", "orange"))
    if services.need_model(det):
        return
    prods = db.list_products()
    c1, c2 = st.columns([2, 1])
    with c2:
        pid = st.selectbox("Link to product (optional)", [ADHOC] + prods.product_id.tolist() if not prods.empty else [ADHOC])
        st.caption(f"Threshold **{config.effective_conf(s):.2f}** ({s['inspection_mode']}) · image size **{det.imgsz}**")
    with c1:
        tab_up, tab_cam = st.tabs(["Upload image(s)", "Capture from camera"])
        files = tab_up.file_uploader("Images", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"], accept_multiple_files=True, label_visibility="collapsed")
        shot = tab_cam.camera_input("Take a picture") if tab_cam.toggle("Enable camera capture", False) else None
    items = [(f.name, f.getvalue()) for f in files or []] + ([("camera_capture.jpg", shot.getvalue())] if shot else [])
    if st.button("Run inspection", type="primary", disabled=not items):
        results = []
        bar = st.progress(0.0, "Running inference...")
        for n, (name, data) in enumerate(items):
            try:
                img = decode_image(data)
                r = pipeline.run_inspection(det, img, s, None if pid == ADHOC else pid, "image", hub=services.hub())
                r["image"], r["name"] = img, name
                results.append(r)
            except Exception as e:
                st.error(f"{name}: {e}")
            bar.progress((n + 1) / len(items))
        bar.empty()
        st.session_state["img_results"] = results
        st.session_state.pop("pdf_img", None)
    results = st.session_state.get("img_results")
    if not results:
        ui.empty("No inspection run yet", "Upload an image above and press Run inspection.")
        return
    for r in results:
        with st.expander(f"{r['name']} - {r['status']}  (#{r['inspection_id']})", expanded=len(results) == 1):
            render_result(r, r["image"])
    pdf_button([r["inspection_id"] for r in results], "img", "Image Inspection Report")
