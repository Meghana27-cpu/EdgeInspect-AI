import pandas as pd
import plotly.express as px
import streamlit as st
import config
from modules import database as db, reporting, ui
from views._common import pdf_button


def render():
    ui.header("Analytics & Reports", "Production-quality analytics from stored inspection data")
    rng = st.date_input("Date range", value=[])
    d1 = rng[0] if len(rng) > 0 else None
    d2 = rng[1] if len(rng) > 1 else d1
    df = db.query_inspections(date_from=d1, date_to=d2, limit=100000)
    if df.empty:
        ui.empty("No data for this period", "Charts are generated only from real inspection records.")
        return
    n, fails = len(df), int((df.status == "FAIL").sum())
    ui.kpi_row([("Total inspected", n, "in selected period", "blue"), ("Pass rate", f"{(n - fails) / n * 100:.1f}%", f"{n - fails} passed", "green"),
                ("Failure rate", f"{fails / n * 100:.1f}%", f"{fails} failed", "red"), ("Defects found", int(df.num_defects.sum()), "individual detections", "orange")])
    st.write("")
    dd = db.all_detections()
    dd = dd[dd.inspection_id.isin(df.id)] if not dd.empty else dd
    c1, c2, c3 = st.columns(3)
    ui.show_plot(ui.style_fig(px.pie(df.status.value_counts().reset_index(), names="status", values="count", hole=.6, title="Pass / Fail",
                                     color="status", color_discrete_map={"PASS": "#22C55E", "FAIL": "#EF4444"})), "a_pf", c1)
    if dd.empty:
        c2.info("No detections in period"); c3.info("No detections in period")
    else:
        dd["Defect"] = dd.defect_class.map(lambda x: config.PRETTY.get(x, x))
        cat = dd.Defect.value_counts().reset_index()
        ui.show_plot(ui.style_fig(px.pie(cat, names="Defect", values="count", title="Defect distribution"), height=320), "a_dist", c2)
        ui.show_plot(ui.style_fig(px.bar(cat, x="Defect", y="count", title="Defects by category"), legend=False), "a_cat", c3)
        heat = dd.groupby(["Defect", "severity"]).size().reset_index(name="n")
        ui.show_plot(ui.style_fig(px.density_heatmap(heat, x="severity", y="Defect", z="n", title="Defect class × severity (size-based)",
                                                      category_orders={"severity": ["Low", "Medium", "High"]}, color_continuous_scale="Blues"), legend=False), "a_heat")
    df["day"] = pd.to_datetime(df.timestamp).dt.date
    daily = df.groupby("day").agg(inspected=("id", "count"), failed=("status", lambda s: (s == "FAIL").sum())).reset_index()
    daily["Failure %"] = daily.failed / daily.inspected * 100
    ui.show_plot(ui.style_fig(px.line(daily, x="day", y="Failure %", markers=True, title="Production quality trend (daily failure %)")), "a_trend")
    prods = db.list_products()
    if not prods.empty:
        j = df.merge(prods[["product_id", "production_line"]], on="product_id", how="inner")
        if not j.empty:
            g = j.groupby("production_line").agg(n=("id", "count"), fail=("status", lambda s: (s == "FAIL").sum())).reset_index()
            g["Failure %"] = g.fail / g.n * 100
            ui.show_plot(ui.style_fig(px.bar(g, x="production_line", y="Failure %", title="Failure rate by production line"), legend=False), "a_line")
    st.markdown("#### Export")
    a, b, c = st.columns([1, 1, 3])
    a.download_button("Inspections CSV", reporting.csv_bytes(df.drop(columns=["day"])), "inspection_report.csv", "text/csv")
    b.download_button("Detections CSV", reporting.csv_bytes(dd.drop(columns=["Defect"], errors="ignore")) if not dd.empty else b"", "detections.csv", "text/csv", disabled=dd.empty)
    with c:
        pdf_button(df.id.head(25).tolist(), "analytics", "Quality Report", "PDF report (latest 25 inspections in period)")
