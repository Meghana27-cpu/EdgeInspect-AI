import pandas as pd
import streamlit as st
import config
from modules import database as db, pipeline, services, ui
from modules.preprocessing import decode_image
from views._common import pdf_button, render_result


def render():
    det, s = services.detector(), services.settings()
    ui.header("Multi-View Inspection", "Inspect several views of one registered product and combine the verdicts")
    if services.need_model(det):
        return
    prods = db.list_products()
    if prods.empty:
        ui.empty("Register a product first", "Multi-view inspections must be linked to a Product ID (Product Registration page).")
        return
    pid = st.selectbox("Product ID", prods.product_id.tolist())
    p = db.get_product(pid)
    ui.status_rows([("Name", p["name"], "blue"), ("Category", p["category"], "gray"), ("Batch", p["batch"], "gray"), ("Line", p["production_line"], "gray")])
    views = st.multiselect("Views to inspect", config.VIEW_NAMES, default=["Front", "Back"])
    uploads = {}
    cols = st.columns(3)
    for i, v in enumerate(views):
        uploads[v] = cols[i % 3].file_uploader(f"{v} view", type=["jpg", "jpeg", "png", "bmp"], key=f"mv_{v}")
    ready = {v: f for v, f in uploads.items() if f}
    if st.button("Run multi-view inspection", type="primary", disabled=not ready):
        sid, out = pipeline.new_session_id(), []
        bar = st.progress(0.0)
        for n, (v, f) in enumerate(ready.items()):
            try:
                img = decode_image(f.getvalue())
                r = pipeline.run_inspection(det, img, s, pid, "multiview", v, sid, hub=services.hub())
                r["image"] = img
                out.append(r)
            except Exception as e:
                st.error(f"{v}: {e}")
            bar.progress((n + 1) / len(ready))
        bar.empty()
        st.session_state["mv"] = {"sid": sid, "pid": pid, "results": out}
        st.session_state.pop("pdf_mv", None)
    mv = st.session_state.get("mv")
    if not mv or not mv["results"]:
        ui.empty("No multi-view result yet", "Choose views, upload one image per view and run the inspection.")
        return
    res = mv["results"]
    n_fail = sum(r["status"] == "FAIL" for r in res)
    overall = "FAIL" if n_fail else "PASS"
    st.markdown(f"### Combined result - product `{mv['pid']}`")
    a, b, c = st.columns([1.3, 1, 1])
    with a:
        ui.verdict(overall)
    b.markdown(ui.kpi("Views failing", f"{n_fail} / {len(res)}", "product fails if any view fails", "red" if n_fail else "green"), unsafe_allow_html=True)
    c.markdown(ui.kpi("Total detections", sum(len(r["result"].detections) for r in res), "sum of per-image detections", "orange"), unsafe_allow_html=True)
    ui.note("Detections are counted <b>per image</b>. This system does <b>not</b> match or track defects across views, so the same physical "
            "defect seen from two views is counted twice. The product-level verdict is simply FAIL if any view fails.")
    ui.show_df(pd.DataFrame([{"View": r["view_name"], "Defects": len(r["result"].detections),
                              "Max confidence": round(max((d.conf for d in r["result"].detections), default=0), 3) or "–",
                              "Inference (ms)": round(r["result"].inference_ms, 1), "Result": r["status"]} for r in res]))
    for r in res:
        with st.expander(f"{r['view_name']} view - {r['status']}"):
            render_result(r, r["image"])
    pdf_button([r["inspection_id"] for r in res], "mv", f"Multi-View Report - {mv['pid']}", "Generate combined PDF report")
