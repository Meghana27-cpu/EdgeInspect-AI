import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import config
from modules import database as db, services, ui


def render():
    det, hub = services.detector(), services.hub()
    ui.header("Overview Dashboard", "Quality-control summary computed from stored inspection records")
    k = db.kpis()
    inf = k["avg_inference_ms"]
    ui.kpi_row([("Products Inspected", f"{k['total']:,}", "stored inspections", "blue"),
                ("Defects Detected", f"{k['defects']:,}", "individual detections", "red"),
                ("Passed", f"{k['passed']:,}", "no defect above threshold", "green"),
                ("Failed", f"{k['failed']:,}", "≥1 defect above threshold", "red")])
    st.write("")
    ui.kpi_row([("Defect Rate", f"{k['fail_rate']:.1f}%" if k["total"] else "–", "inspections flagged FAIL", "orange"),
                ("Avg Inference", f"{inf:.1f} ms" if inf else "–", f"target < {config.TARGET_LATENCY_MS:.0f} ms (model forward pass)",
                 "green" if inf and inf < config.TARGET_LATENCY_MS else "orange"),
                ("Throughput", f"{k['fps']:.1f} FPS" if k["fps"] else "–", "from last 20 inspections (end-to-end)", "cyan"),
                ("System", "Operational" if det.loaded and db.check_db() else "Attention", "model + database check",
                 "green" if det.loaded and db.check_db() else "orange")])
    st.write("")

    df = db.query_inspections(limit=100000)
    if df.empty:
        ui.empty("No inspections recorded yet", "Run an Image, Multi-View or Live inspection – charts will populate from real records.")
    else:
        df["ts"] = pd.to_datetime(df["timestamp"])
        span_days = (df.ts.max() - df.ts.min()).days
        freq, flabel = ("h", "hour") if span_days < 3 else ("D", "day")
        c1, c2 = st.columns([1.6, 1])
        g = df.set_index("ts").groupby([pd.Grouper(freq=freq), "status"]).size().reset_index(name="n")
        f = px.bar(g, x="ts", y="n", color="status", title=f"Inspections per {flabel}", barmode="stack",
                   color_discrete_map={"PASS": "#22C55E", "FAIL": "#EF4444"})
        ui.show_plot(ui.style_fig(f), "d_time", c1)
        p = df.status.value_counts().reset_index()
        f = px.pie(p, names="status", values="count", hole=.62, title="Pass vs Fail", color="status",
                   color_discrete_map={"PASS": "#22C55E", "FAIL": "#EF4444"})
        ui.show_plot(ui.style_fig(f), "d_pie", c2)

        dd = db.all_detections()
        c3, c4, c5 = st.columns(3)
        if dd.empty:
            for c in (c3, c4, c5):
                with c:
                    ui.empty("No detections yet")
        else:
            dd["Defect"] = dd.defect_class.map(lambda x: config.PRETTY.get(x, x))
            cat = dd.Defect.value_counts().reset_index()
            ui.show_plot(ui.style_fig(px.bar(cat, x="Defect", y="count", title="Defects by category"), legend=False), "d_cat", c3)
            sev = dd.severity.value_counts().reindex(["Low", "Medium", "High"]).fillna(0).reset_index()
            ui.show_plot(ui.style_fig(px.bar(sev, x="severity", y="count", title="Severity distribution (by size)", color="severity",
                            color_discrete_map={"Low": "#22C55E", "Medium": "#F59E0B", "High": "#EF4444"}), legend=False), "d_sev", c4)
            ui.show_plot(ui.style_fig(px.histogram(dd, x="confidence", nbins=20, title="Model confidence distribution"), legend=False), "d_conf", c5)
        daily = df.assign(day=df.ts.dt.date).groupby("day").agg(total=("id", "count"), fails=("status", lambda s: (s == "FAIL").sum())).reset_index()
        daily["fail_pct"] = daily.fails / daily.total * 100
        fig = go.Figure([go.Bar(x=daily.day, y=daily.total, name="Inspections", marker_color="#2F7CF6"),
                         go.Scatter(x=daily.day, y=daily.fail_pct, name="Fail %", yaxis="y2", line=dict(color="#EF4444", width=3))])
        fig.update_layout(title="Daily inspection trend", yaxis2=dict(overlaying="y", side="right", title="Fail %", gridcolor="rgba(0,0,0,0)"))
        ui.show_plot(ui.style_fig(fig), "d_daily")

    left, right = st.columns([2.2, 1])
    with left:
        st.markdown("#### Recent inspection activity")
        if df.empty:
            ui.empty("Nothing to show")
        else:
            dd = db.all_detections()
            top = dd.sort_values("confidence", ascending=False).drop_duplicates("inspection_id").set_index("inspection_id").defect_class if not dd.empty else pd.Series(dtype=str)
            r = df.head(12).copy()
            r["Detected defect"] = r.id.map(top).map(lambda x: config.PRETTY.get(x, x) if isinstance(x, str) else "–")
            out = pd.DataFrame({"Product ID": r.product_id.fillna("–"), "Inspection time": r.timestamp, "Detected defect": r["Detected defect"],
                                "Confidence": r.max_confidence.map(lambda v: f"{v:.2f}" if pd.notna(v) else "–"),
                                "Result": r.status, "Processing time": r.total_ms.map(lambda v: f"{v:.1f} ms" if pd.notna(v) else "–"),
                                "Status": r.status.map({"FAIL": "Alert raised", "PASS": "Cleared"})})
            ui.show_df(out)
    with right:
        st.markdown("#### System status")
        snap = hub.snapshot()
        cam = st.session_state.get("camera_status", "Not checked")
        ui.status_rows([
            ("AI Model", "Loaded" if det.loaded else "Not loaded", "green" if det.loaded else "red"),
            ("Edge Runtime", ("Active · " + det.backend) if det.loaded else "Inactive", "green" if det.loaded else "gray"),
            ("Camera", cam, "green" if cam == "Connected" else "orange" if cam == "Not checked" else "red"),
            ("IoT Connection", snap["mode"] if snap["running"] else "Stopped", ("green" if snap["mode"].startswith("MQTT") else "blue") if snap["running"] else "gray"),
            ("Database", "Connected" if db.check_db() else "Disconnected", "green" if db.check_db() else "red")])
        if not det.loaded:
            ui.note(det.error, warn=True)
