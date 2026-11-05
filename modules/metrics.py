"""Model metrics: load training metrics, and evaluate the *deployed* detector against YOLO labels."""
from __future__ import annotations
import json
from pathlib import Path
import cv2
import numpy as np
import config
from modules.preprocessing import list_images


def iou_xyxy(a, b) -> float:
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def read_yolo_labels(path: Path, w: int, h: int):
    out = []
    if path.exists():
        for line in path.read_text().splitlines():
            p = line.split()
            if len(p) >= 5:
                c, cx, cy, bw, bh = int(p[0]), *map(float, p[1:5])
                out.append((c, ((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h)))
    return out


def prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0, "tp": tp, "fp": fp, "fn": fn}


def evaluate_detector(detector, images_dir, labels_dir, conf=0.25, iou_nms=0.45, iou_match=0.5,
                      max_images=None, progress=None) -> dict:
    """Greedy IoU matching per image. Returns object-level P/R/F1 (per class + micro), image-level
    recall (images with >=1 GT defect where >=1 detection was produced), confusion matrix and latency."""
    nc = len(detector.names)
    imgs = list_images(images_dir)[:max_images]
    if not imgs:
        raise ValueError(f"No images found in {images_dir}")
    tp, fp, fn = np.zeros(nc, int), np.zeros(nc, int), np.zeros(nc, int)
    cm = np.zeros((nc + 1, nc + 1), int)          # rows = ground truth, cols = prediction, last = background
    img_pos = img_hit = 0
    ag_tp = ag_fp = ag_fn = 0                      # class-agnostic: IoU match only, class ignored
    lat, inf = [], []
    for n, p in enumerate(imgs):
        img = cv2.imread(str(p))
        if img is None:
            continue
        h, w = img.shape[:2]
        gts = read_yolo_labels(Path(labels_dir) / (p.stem + ".txt"), w, h)
        res = detector.predict(img, conf=conf, iou=iou_nms, annotate=False)
        lat.append(res.total_ms); inf.append(res.inference_ms)
        preds = sorted(res.detections, key=lambda d: -d.conf)
        used = set()
        for d in preds:
            best, bj = 0.0, -1
            for j, (c, b) in enumerate(gts):
                if j in used:
                    continue
                v = iou_xyxy(d.box, b)
                if v > best:
                    best, bj = v, j
            if best >= iou_match:
                used.add(bj)
                gc = gts[bj][0]
                cm[gc, d.cls_id] += 1
                if gc == d.cls_id:
                    tp[gc] += 1
                else:
                    fp[d.cls_id] += 1; fn[gc] += 1
            else:
                fp[d.cls_id] += 1; cm[nc, d.cls_id] += 1
        for j, (c, _) in enumerate(gts):
            if j not in used:
                fn[c] += 1; cm[c, nc] += 1
        used_ag = set()
        for d in preds:
            cands = [(iou_xyxy(d.box, b), j) for j, (_, b) in enumerate(gts) if j not in used_ag]
            best = max(cands, default=(0.0, -1))
            if best[0] >= iou_match:
                used_ag.add(best[1]); ag_tp += 1
            else:
                ag_fp += 1
        ag_fn += len(gts) - len(used_ag)
        if gts:
            img_pos += 1; img_hit += bool(preds)
        if progress:
            progress((n + 1) / len(imgs))
    per = {detector.names[i]: prf(int(tp[i]), int(fp[i]), int(fn[i])) for i in range(nc)}
    overall = prf(int(tp.sum()), int(fp.sum()), int(fn.sum()))
    lat_a = np.array(lat) if lat else np.zeros(1)
    result = {
        "model": detector.model_name, "backend": detector.backend, "images": len(imgs), "conf": conf,
        "iou_match": iou_match, "overall": overall, "per_class": per, "class_agnostic": prf(ag_tp, ag_fp, ag_fn),
        "image_level_recall": img_hit / img_pos if img_pos else None, "images_with_defects": img_pos,
        "confusion_matrix": cm.tolist(), "labels": list(detector.names) + ["background"],
        "latency_ms": {"mean": float(lat_a.mean()), "p50": float(np.percentile(lat_a, 50)),
                       "p95": float(np.percentile(lat_a, 95)), "inference_mean": float(np.mean(inf)) if inf else 0.0},
        "timestamp": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
    }
    return result


def save_json(obj, name) -> Path:
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    p = config.RESULTS_DIR / name
    p.write_text(json.dumps(obj, indent=2))
    return p


def load_json(name):
    try:
        return json.loads((config.RESULTS_DIR / name).read_text())
    except (OSError, ValueError):
        return None
