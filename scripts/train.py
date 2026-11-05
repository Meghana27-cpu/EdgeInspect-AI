"""Train YOLO26n on NEU-DET, validate, export to ONNX and write metrics for the dashboard.

Usage:
    python scripts/train.py --data datasets/neu-det/data.yaml --epochs 100 --imgsz 640 --device 0
Requires:  pip install ultralytics   (a release that includes YOLO26; the first run downloads yolo26n.pt)
"""
from __future__ import annotations
import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from modules import metrics


def box_metrics(m, split, weights):
    b = m.box
    names = m.names
    per = {}
    f1 = getattr(b, "f1", None)
    for k, ci in enumerate(b.ap_class_index):
        per[names[int(ci)]] = {"precision": float(b.p[k]), "recall": float(b.r[k]), "f1": float(f1[k]) if f1 is not None else 0.0,
                               "map50": float(b.ap50[k]), "map50_95": float(b.ap[k])}
    p, r = float(b.mp), float(b.mr)
    return {"split": split, "weights": str(weights), "timestamp": datetime.now().isoformat(timespec="seconds"),
            "overall": {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0,
                        "map50": float(b.map50), "map50_95": float(b.map)}, "per_class": per}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="datasets/neu-det/data.yaml")
    ap.add_argument("--model", default="yolo26n.pt")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=None)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--name", default="neudet_yolo26n")
    ap.add_argument("--no-export", action="store_true")
    a = ap.parse_args()
    from ultralytics import YOLO

    model = YOLO(a.model)
    model.train(data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, device=a.device, patience=a.patience,
                project=str(config.BASE_DIR / "runs"), name=a.name, exist_ok=True, plots=True)
    save_dir = Path(model.trainer.save_dir)
    best = save_dir / "weights" / "best.pt"
    config.MODELS_DIR.mkdir(exist_ok=True)
    shutil.copy2(best, config.MODELS_DIR / "best.pt")
    config.RESULTS_DIR.mkdir(exist_ok=True)
    for f in ("results.png",):
        if (save_dir / f).exists():
            shutil.copy2(save_dir / f, config.RESULTS_DIR / f)

    best_model = YOLO(str(best))
    for split in ("val", "test"):
        try:
            m = best_model.val(data=a.data, split=split, imgsz=a.imgsz, plots=True, project=str(config.BASE_DIR / "runs"),
                               name=f"{a.name}_{split}", exist_ok=True)
        except Exception as e:           # e.g. no test split configured
            print(f"[skip] {split} validation: {e}"); continue
        rec = box_metrics(m, split, config.MODELS_DIR / "best.pt")
        rec["images"] = int(getattr(m, "seen", 0))
        metrics.save_json(rec, "training_metrics.json" if split == "val" else "test_metrics.json")
        cmn = Path(m.save_dir) / "confusion_matrix_normalized.png"
        if split == "val" and cmn.exists():
            shutil.copy2(cmn, config.RESULTS_DIR / cmn.name)
        o = rec["overall"]
        print(f"{split}: P={o['precision']:.3f} R={o['recall']:.3f} mAP50={o['map50']:.3f} mAP50-95={o['map50_95']:.3f}")
    if not a.no_export:
        out = Path(best_model.export(format="onnx", imgsz=a.imgsz, simplify=True, opset=17))
        shutil.copy2(out, config.MODELS_DIR / "best.onnx")
        print("Exported", config.MODELS_DIR / "best.onnx")


if __name__ == "__main__":
    main()
