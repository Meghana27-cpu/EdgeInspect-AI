import re
import streamlit as st
import config
from modules import database as db, ui


def render():
    ui.header("Product Registration", "Register products so inspections can be traced to a Product ID")
    with st.form("reg", clear_on_submit=False):
        c1, c2 = st.columns(2)
        pid = c1.text_input("Product ID *", placeholder="e.g. STL-2026-0001")
        c3, c4, c5 = st.columns(3)
        cat = c3.selectbox("Category", config.PRODUCT_CATEGORIES)
        batch = c4.text_input("Batch number *")
        line = c5.text_input("Production line *", placeholder="Line-02")
        desc = st.text_area("Description (optional)", height=80)
        ok = st.form_submit_button("Register product", type="primary")
    if ok:
        pid = pid.strip()
        if not (pid and batch.strip() and line.strip()):
            st.error("Product ID, name, batch number and production line are required.")
        elif not re.fullmatch(r"[A-Za-z0-9_\-./]{1,64}", pid):
            st.error("Product ID may only contain letters, digits and - _ . / (max 64 characters).")
        elif db.add_product(pid, cat, batch, line, desc):
            st.success(f"Product {pid} registered.")
        else:
            st.error(f"Product ID '{pid}' already exists. Product IDs must be unique.")
    st.markdown("#### Registered products")
    df = db.list_products()
    if df.empty:
        ui.empty("No products registered yet")
    else:
        df = df.rename(columns={"product_id": "Product ID", "category": "Category", "batch": "Batch",
                                "production_line": "Line", "description": "Description", "created_at": "Registered",
                                "inspections": "Inspections", "failed": "Failed"})
        ui.show_df(df)
