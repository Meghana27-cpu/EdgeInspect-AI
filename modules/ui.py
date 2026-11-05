"""Shared Streamlit UI helpers: theme, cards, pills, Plotly styling, version-safe wrappers."""
from __future__ import annotations
import html
import cv2
import numpy as np
import streamlit as st
import config

TONES = {"green": "#22C55E", "red": "#EF4444", "orange": "#F59E0B", "blue": "#2F7CF6", "cyan": "#22D3EE"}
SEQ = ["#2F7CF6", "#22D3EE", "#8B5CF6", "#F59E0B", "#22C55E", "#EF4444"]


def load_css():
    st.markdown(f"<style>{(config.BASE_DIR / 'assets/css/theme.css').read_text()}</style>", unsafe_allow_html=True)


def header(title: str, sub: str = "", badge: str = ""):
    st.markdown(f"<div class='pagehead'><div><h1>{html.escape(title)}</h1><p>{html.escape(sub)}</p></div>"
                f"<div>{badge}</div></div>", unsafe_allow_html=True)


def pill(text: str, tone: str = "blue") -> str:
    return f"<span class='pill {tone}'>{html.escape(str(text))}</span>"


def status_pill(status: str) -> str:
    return pill(status, {"PASS": "green", "FAIL": "red"}.get(status, "gray"))


def kpi(label, value, sub="", tone="blue") -> str:
    return (f"<div class='kpi {tone}'><div class='l'>{html.escape(label)}</div><div class='v'>{html.escape(str(value))}</div>"
            f"<div class='s'>{html.escape(sub)}</div></div>")


def kpi_row(items: list[tuple]):
    for col, it in zip(st.columns(len(items)), items):
        col.markdown(kpi(*it), unsafe_allow_html=True)


def empty(msg: str, hint: str = ""):
    st.markdown(f"<div class='empty'><b>{html.escape(msg)}</b><br><span style='font-size:.85rem'>{html.escape(hint)}</span></div>",
                unsafe_allow_html=True)


def note(msg: str, warn: bool = False):
    st.markdown(f"<div class='note {'warn' if warn else ''}'>{msg}</div>", unsafe_allow_html=True)


def verdict(status: str):
    st.markdown(f"<div class='verdict {status.lower()}'>{status}</div>", unsafe_allow_html=True)


def card_open(title: str):
    st.markdown(f"<div class='card'><h4>{html.escape(title)}</h4>", unsafe_allow_html=True)


def card_close():
    st.markdown("</div>", unsafe_allow_html=True)


def status_rows(rows: list[tuple[str, str, str]]):
    st.markdown("".join(f"<div class='statusrow'><span>{html.escape(a)}</span>{pill(b, t)}</div>" for a, b, t in rows),
                unsafe_allow_html=True)


def style_fig(fig, height=320, legend=True):
    fig.update_layout(height=height, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      margin=dict(l=10, r=10, t=40, b=10), colorway=SEQ, showlegend=legend,
                      font=dict(color="#C9D6EA", size=12), title_font=dict(size=14, color="#E6EDF7"),
                      legend=dict(orientation="h", y=-0.2))
    fig.update_xaxes(gridcolor="#1B2A44", zeroline=False)
    fig.update_yaxes(gridcolor="#1B2A44", zeroline=False)
    return fig


# --- wrappers that work across Streamlit versions (use_container_width -> width="stretch") ---
def show_plot(fig, key=None, where=st):
    try:
        where.plotly_chart(fig, width="stretch", key=key)
    except TypeError:
        where.plotly_chart(fig, use_container_width=True, key=key)


def show_df(df, **kw):
    try:
        st.dataframe(df, width="stretch", hide_index=True, **kw)
    except TypeError:
        st.dataframe(df, use_container_width=True, hide_index=True, **kw)


def show_img(img_bgr_or_path, caption=None):
    img = img_bgr_or_path
    if isinstance(img, str):
        p = config.resolve(img)
        if not p.exists():
            st.caption("image file not found"); return
        img = cv2.imread(str(p))
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if isinstance(img, np.ndarray) else img
    try:
        st.image(rgb, caption=caption, width="stretch")
    except TypeError:
        st.image(rgb, caption=caption, use_container_width=True)
