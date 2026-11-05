"""Image decoding, letterbox preprocessing and dataset validation."""
from __future__ import annotations
import time
from pathlib import Path
import cv2
import numpy as np

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def decode_image(data: bytes) -> np.ndarray:
    """Decode bytes to a 3-channel BGR image. NEU-DET images are grayscale, so we expand them."""
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("File could not be decoded as an image")
    return img


def letterbox(img: np.ndarray, size: int, color=(114, 114, 114)):
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR) if (nh, nw) != (h, w) else img
    top, left = (size - nh) // 2, (size - nw) // 2
    out = cv2.copyMakeBorder(resized, top, size - nh - top, left, size - nw - left,
                             cv2.BORDER_CONSTANT, value=color)
    return out, r, (left, top)


def to_tensor(img_bgr: np.ndarray, size: int, fp16: bool = False):
    """BGR image -> (1,3,size,size) tensor in [0,1], plus (ratio, pad)."""
    lb, r, pad = letterbox(img_bgr, size)
    x = np.ascontiguousarray(lb[:, :, ::-1].transpose(2, 0, 1), dtype=np.float32) / 255.0
    return x[None].astype(np.float16 if fp16 else np.float32), r, pad


def list_images(folder) -> list[Path]:
    p = Path(folder)
    if not p.is_dir():
        return []
    return sorted(f for f in p.rglob("*") if f.suffix.lower() in IMG_EXT)


def validate_yolo_dataset(root, class_names: list[str]) -> dict:
    """Check a YOLO-format dataset (images/{train,val}, labels/{train,val})."""
    root = Path(root)
    rep = {"root": str(root), "ok": False, "splits": {}, "problems": []}
    if not root.is_dir():
        rep["problems"].append("Dataset folder does not exist")
        return rep
    for split in ("train", "val"):
        imgs = list_images(root / "images" / split)
        n_missing = n_bad = n_boxes = 0
        counts = {i: 0 for i in range(len(class_names))}
        for im in imgs:
            lab = root / "labels" / split / (im.stem + ".txt")
            if not lab.exists():
                n_missing += 1
                continue
            for line in lab.read_text().splitlines():
                parts = line.split()
                try:
                    c = int(parts[0]); vals = list(map(float, parts[1:5]))
                    if c not in counts or len(vals) != 4 or any(v < 0 or v > 1 for v in vals):
                        raise ValueError
                    counts[c] += 1; n_boxes += 1
                except (ValueError, IndexError):
                    n_bad += 1
        rep["splits"][split] = {"images": len(imgs), "missing_labels": n_missing,
                                "invalid_lines": n_bad, "boxes": n_boxes,
                                "per_class": {class_names[i]: v for i, v in counts.items()}}
        if not imgs:
            rep["problems"].append(f"No images found in images/{split}")
        if n_missing:
            rep["problems"].append(f"{n_missing} {split} images have no label file")
        if n_bad:
            rep["problems"].append(f"{n_bad} invalid label lines in {split}")
    rep["ok"] = not rep["problems"]
    return rep


class Timer:
    def __enter__(self):
        self.t = time.perf_counter(); return self
    def __exit__(self, *a):
        self.ms = (time.perf_counter() - self.t) * 1000
