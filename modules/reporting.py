"""CSV and PDF report generation."""
from __future__ import annotations
from datetime import datetime
from pathlib import Path
import pandas as pd
import config
from modules import database as db


def csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def _img(rel, max_w, max_h):
    from reportlab.platypus import Image, Paragraph
    from reportlab.lib.styles import getSampleStyleSheet
    p = config.resolve(rel) if rel else None
    if not p or not p.exists():
        return Paragraph("(image not available)", getSampleStyleSheet()["Italic"])
    from PIL import Image as PI
    with PI.open(p) as im:
        w, h = im.size
    s = min(max_w / w, max_h / h)
    return Image(str(p), width=w * s, height=h * s)


def build_pdf(inspection_ids: list[int], out_path=None, title="Inspection Report", db_path=None) -> Path:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = Path(out_path or config.REPORTS_DIR / f"report_{datetime.now():%Y%m%d_%H%M%S}.pdf")
    ss = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=ss["Title"], fontSize=16, textColor=colors.HexColor("#0B1F3A"))
    small = ParagraphStyle("sm", parent=ss["Normal"], fontSize=8, textColor=colors.grey)
    doc = SimpleDocTemplate(str(out), pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm, topMargin=15 * mm, bottomMargin=15 * mm,
                            title=title)
    story = [Paragraph(config.APP_NAME + " - " + title, h1),
             Paragraph(config.PROJECT_NAME, small),
             Paragraph(f"Generated {datetime.now():%Y-%m-%d %H:%M:%S}", small), Spacer(1, 6 * mm)]
    rows = [db.get_inspection(i, db_path) for i in inspection_ids]
    rows = [r for r in rows if r]
    if not rows:
        story.append(Paragraph("No inspections selected.", ss["Normal"]))
    if len(rows) > 1:
        fails = sum(r["status"] == "FAIL" for r in rows)
        story += [Paragraph(f"<b>Summary:</b> {len(rows)} inspection(s), {len(rows) - fails} PASS, {fails} FAIL. "
                            "Each image is judged independently; detections are not matched across views.", ss["Normal"]),
                  Spacer(1, 4 * mm)]
    for n, r in enumerate(rows):
        prod = db.get_product(r["product_id"], db_path) if r["product_id"] else None
        colr = colors.HexColor("#1E9E5A" if r["status"] == "PASS" else "#D93636")
        info = [["Inspection ID", r["id"], "Result", r["status"]],
                ["Product ID", r["product_id"] or "-", "Timestamp", r["timestamp"]],
                ["Product name", (prod or {}).get("name", "-"), "Batch / Line", f"{(prod or {}).get('batch', '-')} / {(prod or {}).get('production_line', '-')}"],
                ["View", r["view_name"] or "-", "Defects", r["num_defects"]],
                ["Inference (model)", f"{r['inference_ms']:.1f} ms" if r["inference_ms"] is not None else "-",
                 "Total latency", f"{r['total_ms']:.1f} ms" if r["total_ms"] is not None else "-"],
                ["Model / backend", f"{r['model_name']} / {r['backend']}", "Conf. threshold", r["threshold"]]]
        t = Table(info, colWidths=[32 * mm, 52 * mm, 32 * mm, 52 * mm])
        t.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, -1), 8), ("GRID", (0, 0), (-1, -1), .25, colors.lightgrey),
                               ("BACKGROUND", (0, 0), (0, -1), colors.whitesmoke), ("BACKGROUND", (2, 0), (2, -1), colors.whitesmoke),
                               ("TEXTCOLOR", (3, 0), (3, 0), colr), ("FONTNAME", (3, 0), (3, 0), "Helvetica-Bold")]))
        imgs = Table([[_img(r["original_path"], 85 * mm, 70 * mm), _img(r["annotated_path"], 85 * mm, 70 * mm)],
                      ["Original", "Annotated"]], colWidths=[90 * mm, 90 * mm])
        imgs.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER"), ("FONTSIZE", (0, 1), (-1, 1), 8)]))
        d = db.get_detections(r["id"], db_path)
        dt = [["Defect class", "Confidence", "Severity (by size)", "Size share"]] + \
             [[config.PRETTY.get(x.defect_class, x.defect_class), f"{x.confidence:.3f}", x.severity, f"{x.area_ratio * 100:.2f}%"]
              for x in d.itertuples()] if len(d) else [["No defects above threshold"]]
        dtab = Table(dt, colWidths=[55 * mm, 35 * mm, 45 * mm, 35 * mm] if len(d) else None)
        dtab.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, -1), 8), ("GRID", (0, 0), (-1, -1), .25, colors.lightgrey),
                                  ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8EEF7"))]))
        story += [Paragraph(f"Inspection #{r['id']}", ss["Heading3"]), t, Spacer(1, 3 * mm), imgs, Spacer(1, 3 * mm), dtab]
        story.append(Paragraph("Confidence = model certainty; severity = defect size relative to image. They are independent.", small))
        if n < len(rows) - 1:
            story.append(PageBreak())
    doc.build(story)
    return out
