import streamlit as st
import config
from modules import database as db, reporting, ui
from views._common import pdf_button


def render():
    ui.header("Inspection History", "Search, filter and review stored inspections")
    c = st.columns([1.2, 1.4, 1.2, 1, 1])
    pid = c[0].text_input("Product ID contains")
    rng = c[1].date_input("Date range", value=[])
    defect = c[2].selectbox("Defect type", ["All"] + config.CLASS_NAMES, format_func=lambda x: config.PRETTY.get(x, x))
    status = c[3].selectbox("Result", ["All", "PASS", "FAIL"])
    itype = c[4].selectbox("Type", ["All", "image", "multiview", "live"])
    d1 = rng[0] if len(rng) > 0 else None
    d2 = rng[1] if len(rng) > 1 else d1
    df = db.query_inspections(pid, d1, d2, defect, status, itype)
    if df.empty:
        ui.empty("No inspections match the filters", "Run an inspection or relax the filters.")
        return
    show = df[["id", "timestamp", "product_id", "inspection_type", "view_name", "status", "num_defects", "max_confidence", "inference_ms", "total_ms", "model_name"]]
    st.caption(f"{len(df)} record(s)")
    ui.show_df(show)
    a, b = st.columns([1, 4])
    a.download_button("Export CSV", reporting.csv_bytes(df), "inspections.csv", "text/csv")
    with b:
        pdf_button(df.id.head(25).tolist(), "hist_all", "Inspection History Report", "PDF of filtered records (first 25)")
    st.markdown("#### Inspection detail")
    iid = st.selectbox("Inspection ID", df.id.tolist())
    r = db.get_inspection(iid)
    l, rr = st.columns(2)
    with l:
        st.caption("Original"); ui.show_img(r["original_path"])
    with rr:
        st.caption("Annotated"); ui.show_img(r["annotated_path"])
    st.markdown(ui.status_pill(r["status"]) + "  " + ui.pill(r["timestamp"], "gray") + "  " + ui.pill(r["backend"] or "-", "blue"), unsafe_allow_html=True)
    d = db.get_detections(iid)
    if d.empty:
        st.success("No defects were detected in this inspection.")
    else:
        d["defect_class"] = d.defect_class.map(lambda x: config.PRETTY.get(x, x))
        ui.show_df(d[["defect_class", "confidence", "severity", "area_ratio", "x1", "y1", "x2", "y2"]])
    pdf_button([iid], f"hist_{iid}", f"Inspection #{iid}", "PDF report for this inspection")
