"""Inspection pipeline: run detector, persist images + DB rows, raise alerts. No UI dependencies."""
from __future__ import annotations
import uuid
from datetime import datetime
import cv2
import numpy as np
import config
from modules import database as db


def new_session_id() -> str:
    return "S" + datetime.now().strftime("%Y%m%d%H%M%S") + uuid.uuid4().hex[:4].upper()


def decide(detections: list) -> str:
    """PASS = no detection above the configured threshold; FAIL = at least one."""
    return "FAIL" if detections else "PASS"


def save_image(img: np.ndarray, folder, stem: str) -> str:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{stem}.jpg"
    cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return str(path.relative_to(config.BASE_DIR))


def run_inspection(detector, img: np.ndarray, settings: dict, product_id: str | None = None,
                   inspection_type: str = "image", view_name: str | None = None,
                   session_id: str | None = None, hub=None, db_path=None, persist: bool = True) -> dict:
    conf = config.effective_conf(settings)
    res = detector.predict(img, conf=conf, iou=settings["iou_threshold"])
    return persist_result(detector, img, res, conf, product_id, inspection_type, view_name, session_id,
                          hub, db_path, persist)


def persist_result(detector, img, res, conf, product_id=None, inspection_type="image", view_name=None,
                   session_id=None, hub=None, db_path=None, persist=True) -> dict:
    """Store an already computed InferenceResult (images + DB rows + alert)."""
    status = decide(res.detections)
    out = {"status": status, "result": res, "conf_threshold": conf, "product_id": product_id,
           "view_name": view_name, "session_id": session_id, "inspection_id": None,
           "original_path": None, "annotated_path": None}
    if not persist:
        return out
    stem = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out["original_path"] = save_image(img, config.UPLOAD_DIR, "orig_" + stem)
    out["annotated_path"] = save_image(res.annotated, config.RESULTS_DIR, "ann_" + stem)
    dets = [{"defect_class": d.label, "confidence": d.conf, "severity": d.severity,
             "area_ratio": d.area_ratio, "box": d.box} for d in res.detections]
    rec = {"product_id": product_id, "session_id": session_id, "inspection_type": inspection_type,
           "view_name": view_name, "original_path": out["original_path"], "annotated_path": out["annotated_path"],
           "status": status, "num_defects": len(dets), "max_confidence": max((d["confidence"] for d in dets), default=None),
           "inference_ms": res.inference_ms, "total_ms": res.total_ms, "model_name": detector.model_name,
           "backend": detector.backend, "threshold": conf}
    out["inspection_id"] = db.add_inspection(rec, dets, db_path)
    if status == "FAIL":
        worst = max(res.detections, key=lambda d: d.conf)
        high = any(d.severity == "High" for d in res.detections)
        msg = (f"{len(dets)} defect(s) on {product_id or 'unregistered item'}"
               f"{' [' + view_name + ']' if view_name else ''}: top = "
               f"{config.PRETTY.get(worst.label, worst.label)} ({worst.conf:.2f})")
        db.add_alert("inspection", "critical" if high else "warning", msg, product_id, db_path)
        if hub is not None:
            hub.publish_event("alerts/defect", {"product_id": product_id, "defects": len(dets),
                                                "top": worst.label, "confidence": round(worst.conf, 3)})
    return out
