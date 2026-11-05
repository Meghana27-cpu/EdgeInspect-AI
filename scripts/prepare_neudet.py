"""Convert NEU-DET (Pascal-VOC XML) into a YOLO dataset with a stratified train/val/test split.

Usage:
    python scripts/prepare_neudet.py --src /path/to/NEU-DET --dst datasets/neu-det [--split 0.7 0.15 0.15] [--seed 42]

--src may be the extracted Kaggle folder (train/validation sub-folders are merged and re-split so that a
genuinely held-out test set exists). Images are matched to XML files by file stem anywhere under --src.
"""
from __future__ import annotations
import argparse
import random
import re
import shutil
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

EXT = {".jpg", ".jpeg", ".png", ".bmp"}
CANON = {re.sub(r"[^a-z]", "", c.lower()): i for i, c in enumerate(config.CLASS_NAMES)}


def class_id(name: str) -> int:
    k = re.sub(r"[^a-z]", "", name.lower())
    if k not in CANON:
        raise KeyError(f"Unknown class '{name}'")
    return CANON[k]


def parse_xml(path: Path):
    root = ET.parse(path).getroot()
    size = root.find("size")
    w = int(float(size.findtext("width"))) if size is not None and size.findtext("width") else 0
    h = int(float(size.findtext("height"))) if size is not None and size.findtext("height") else 0
    objs = []
    for o in root.iter("object"):
        bb = o.find("bndbox")
        objs.append((class_id(o.findtext("name", "")), *(float(bb.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))))
    return (root.findtext("filename") or path.stem), w, h, objs


def convert(src: Path, dst: Path, split=(0.7, 0.15, 0.15), seed=42) -> dict:
    images = {}
    for p in src.rglob("*"):
        if p.suffix.lower() in EXT:
            images.setdefault(p.stem, p)
    items, skipped = [], []
    for x in sorted(src.rglob("*.xml")):
        try:
            _, w, h, objs = parse_xml(x)
        except (ET.ParseError, KeyError, ValueError, TypeError) as e:
            skipped.append((x.name, str(e))); continue
        img = images.get(x.stem)
        if img is None or not objs:
            skipped.append((x.name, "no matching image or no objects")); continue
        if not (w and h):
            import cv2
            h, w = cv2.imread(str(img)).shape[:2]
        lines = []
        for c, x1, y1, x2, y2 in objs:
            x1, x2 = max(0, min(x1, w)), max(0, min(x2, w)); y1, y2 = max(0, min(y1, h)), max(0, min(y2, h))
            if x2 - x1 < 1 or y2 - y1 < 1:
                continue
            lines.append(f"{c} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
        if lines:
            items.append((img, lines, objs[0][0]))
    if not items:
        raise SystemExit("No valid image/XML pairs found. Check --src.")
    rng = random.Random(seed)
    by_cls = defaultdict(list)
    for it in items:
        by_cls[it[2]].append(it)
    splits = {"train": [], "val": [], "test": []}
    for c, lst in by_cls.items():          # stratified by the first object's class
        rng.shuffle(lst)
        n_tr, n_va = round(len(lst) * split[0]), round(len(lst) * split[1])
        splits["train"] += lst[:n_tr]; splits["val"] += lst[n_tr:n_tr + n_va]; splits["test"] += lst[n_tr + n_va:]
    if dst.exists():
        shutil.rmtree(dst)
    for name, lst in splits.items():
        (dst / "images" / name).mkdir(parents=True); (dst / "labels" / name).mkdir(parents=True)
        for img, lines, _ in lst:
            shutil.copy2(img, dst / "images" / name / img.name)
            (dst / "labels" / name / (img.stem + ".txt")).write_text("\n".join(lines) + "\n")
    yaml = (f"path: {dst.resolve()}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n"
            + "".join(f"  {i}: {n}\n" for i, n in enumerate(config.CLASS_NAMES)))
    (dst / "data.yaml").write_text(yaml)
    return {"images": len(items), **{k: len(v) for k, v in splits.items()}, "skipped": skipped}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--dst", default=config.DATASETS_DIR / "neu-det", type=Path)
    ap.add_argument("--split", nargs=3, type=float, default=[0.7, 0.15, 0.15])
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    rep = convert(a.src, a.dst, tuple(a.split), a.seed)
    print(f"Converted {rep['images']} images -> train {rep['train']} / val {rep['val']} / test {rep['test']}; skipped {len(rep['skipped'])}")
    for n, why in rep["skipped"][:10]:
        print("  skipped", n, "-", why)
    print("data.yaml written to", a.dst / "data.yaml")
