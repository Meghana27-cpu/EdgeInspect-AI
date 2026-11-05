"""YOLO detector with ONNX Runtime and PyTorch (Ultralytics) back-ends.

Supports both ONNX output layouts:
  * end-to-end / NMS-free (YOLO26 default export): (1, N, 6) = x1,y1,x2,y2,conf,cls
  * legacy raw head (YOLOv8-style): (1, 4+nc, anchors) = cx,cy,w,h,class scores (NMS applied here)
"""
from __future__ import annotations
import ast
import time
from dataclasses import dataclass, field
import cv2
import numpy as np

import config
from modules.preprocessing import to_tensor

PALETTE = [(60, 76, 231), (0, 165, 255), (255, 180, 0), (200, 80, 200), (60, 200, 120), (255, 255, 0)]


@dataclass
class Detection:
    cls_id: int
    label: str
    conf: float
    box: tuple          # x1,y1,x2,y2 in original image pixels
    area_ratio: float   # box area / image area (drives severity, NOT confidence)
    severity: str


@dataclass
class InferenceResult:
    detections: list = field(default_factory=list)
    annotated: np.ndarray | None = None
    preprocess_ms: float = 0.0
    inference_ms: float = 0.0     # model forward pass only
    postprocess_ms: float = 0.0
    total_ms: float = 0.0         # pre + inference + post (drawing excluded)
    shape: tuple = (0, 0)


def severity_from_ratio(ratio: float, medium_pct: float = 2.0, high_pct: float = 8.0) -> str:
    """Severity is based on defect SIZE (share of the image), independent of model confidence."""
    pct = ratio * 100
    return "High" if pct >= high_pct else "Medium" if pct >= medium_pct else "Low"


def decode_output(out, nc: int, conf_thr: float, iou_thr: float):
    """Return list of (x1,y1,x2,y2,conf,cls) in network-input coordinates."""
    p = np.asarray(out, dtype=np.float32)
    if p.ndim == 3:
        p = p[0]
    if p.ndim != 2:
        raise ValueError(f"Unexpected model output shape {np.shape(out)}")
    if p.shape[1] == 6 and p.shape[0] > 6:                      # end-to-end (N,6)
        keep = p[:, 4] >= conf_thr
        return [(*map(float, r[:4]), float(r[4]), int(r[5])) for r in p[keep]]
    if p.shape[0] == 4 + nc:                                     # legacy (4+nc, A)
        p = p.T
    elif p.shape[1] != 4 + nc:
        raise ValueError(f"Output shape {np.shape(out)} does not match {nc} classes")
    scores = p[:, 4:]
    cls = scores.argmax(1)
    conf = scores[np.arange(len(p)), cls]
    keep = conf >= conf_thr
    p, cls, conf = p[keep], cls[keep], conf[keep]
    if not len(p):
        return []
    xywh = np.stack([p[:, 0] - p[:, 2] / 2, p[:, 1] - p[:, 3] / 2, p[:, 2], p[:, 3]], 1)
    idx = np.array(cv2.dnn.NMSBoxes(xywh.tolist(), conf.tolist(), conf_thr, iou_thr)).reshape(-1)
    return [(float(xywh[i, 0]), float(xywh[i, 1]), float(xywh[i, 0] + xywh[i, 2]),
             float(xywh[i, 1] + xywh[i, 3]), float(conf[i]), int(cls[i])) for i in idx]


def draw(img: np.ndarray, dets: list) -> np.ndarray:
    out = img.copy()
    t = max(1, round(min(out.shape[:2]) / 250))
    for d in dets:
        x1, y1, x2, y2 = map(int, d.box)
        col = PALETTE[d.cls_id % len(PALETTE)]
        cv2.rectangle(out, (x1, y1), (x2, y2), col, t + 1)
        txt = f"{config.PRETTY.get(d.label, d.label)} {d.conf:.2f}"
        fs = 0.4 * t + 0.2
        (tw, th), bl = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, fs, t)
        y0 = max(y1, th + bl + 2)
        cv2.rectangle(out, (x1, y0 - th - bl - 2), (x1 + tw + 4, y0), col, -1)
        cv2.putText(out, txt, (x1 + 2, y0 - bl), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), t, cv2.LINE_AA)
    return out


class Detector:
    def __init__(self, model_path, imgsz=640, runtime="auto", threads=0, class_names=None,
                 medium_pct=2.0, high_pct=8.0):
        self.path = config.resolve(model_path)
        self.imgsz = int(imgsz)
        self.runtime = runtime
        self.threads = int(threads)
        self.names = list(class_names or config.CLASS_NAMES)
        self.medium_pct, self.high_pct = medium_pct, high_pct
        self.loaded = False
        self.backend = "none"
        self.providers: list = []
        self.error = ""
        self._sess = self._model = None
        self._fp16 = False
        self.model_name = self.path.name
        self._load()

    def _load(self):
        if not self.path.exists():
            self.error = f"Model file not found: {self.path}"
            return
        try:
            suffix = self.path.suffix.lower()
            if suffix == ".onnx":
                if self.runtime == "pytorch":
                    raise RuntimeError("Runtime is 'pytorch' but the model is an .onnx file")
                self._load_onnx()
            elif suffix == ".pt":
                if self.runtime.startswith("onnxruntime"):
                    raise RuntimeError("Runtime is ONNX Runtime but the model is .pt - export it on the Edge Optimization page")
                self._load_torch()
            else:
                raise RuntimeError(f"Unsupported model type {suffix}")
            self.loaded = True
        except Exception as e:  # never crash the UI
            self.error = f"{type(e).__name__}: {e}"

    def _load_onnx(self):
        import onnxruntime as ort
        avail = ort.get_available_providers()
        order = []
        if self.runtime == "onnxruntime-gpu":
            order = [p for p in ("TensorrtExecutionProvider", "CUDAExecutionProvider") if p in avail]
            if not order:
                raise RuntimeError("onnxruntime-gpu requested but no CUDA/TensorRT provider is available")
        elif self.runtime == "auto" and "CUDAExecutionProvider" in avail:
            order = ["CUDAExecutionProvider"]
        order.append("CPUExecutionProvider")
        so = ort.SessionOptions()
        if self.threads > 0:
            so.intra_op_num_threads = self.threads
        self._sess = ort.InferenceSession(str(self.path), so, providers=order)
        self.providers = self._sess.get_providers()
        self.backend = f"ONNX Runtime ({self.providers[0]})"
        inp = self._sess.get_inputs()[0]
        self._in_name = inp.name
        self._fp16 = "float16" in inp.type
        h = inp.shape[2] if len(inp.shape) == 4 else None
        if isinstance(h, int) and h > 0:
            self.imgsz = h            # static-shape models dictate the size
        meta = self._sess.get_modelmeta().custom_metadata_map or {}
        if "names" in meta:
            try:
                names = ast.literal_eval(meta["names"])
                self.names = [names[i] for i in sorted(names)] if isinstance(names, dict) else list(names)
            except (ValueError, SyntaxError):
                pass

    def _load_torch(self):
        from ultralytics import YOLO
        self._model = YOLO(str(self.path))
        n = self._model.names
        self.names = [n[i] for i in sorted(n)] if isinstance(n, dict) else list(n)
        self.backend = "PyTorch (Ultralytics)"
        self.providers = ["PyTorch"]

    def _make(self, boxes, shape):
        h, w = shape
        dets = []
        for x1, y1, x2, y2, c, k in boxes:
            x1, x2 = sorted((min(max(x1, 0), w), min(max(x2, 0), w)))
            y1, y2 = sorted((min(max(y1, 0), h), min(max(y2, 0), h)))
            if x2 - x1 < 1 or y2 - y1 < 1:
                continue
            ratio = (x2 - x1) * (y2 - y1) / float(w * h)
            label = self.names[k] if 0 <= k < len(self.names) else str(k)
            dets.append(Detection(k, label, float(c), (x1, y1, x2, y2), ratio,
                                  severity_from_ratio(ratio, self.medium_pct, self.high_pct)))
        dets.sort(key=lambda d: -d.conf)
        return dets

    def predict(self, img: np.ndarray, conf: float = 0.25, iou: float = 0.45, annotate: bool = True) -> InferenceResult:
        if not self.loaded:
            raise RuntimeError(self.error or "Model not loaded")
        h, w = img.shape[:2]
        res = InferenceResult(shape=(h, w))
        t0 = time.perf_counter()
        if self._sess is not None:
            x, r, (px, py) = to_tensor(img, self.imgsz, self._fp16)
            t1 = time.perf_counter()
            out = self._sess.run(None, {self._in_name: x})[0]
            t2 = time.perf_counter()
            raw = decode_output(out, len(self.names), conf, iou)
            boxes = [((a - px) / r, (b - py) / r, (c - px) / r, (d - py) / r, s, k) for a, b, c, d, s, k in raw]
            res.detections = self._make(boxes, (h, w))
            t3 = time.perf_counter()
            res.preprocess_ms, res.inference_ms, res.postprocess_ms = (t1 - t0) * 1e3, (t2 - t1) * 1e3, (t3 - t2) * 1e3
        else:
            r0 = self._model.predict(img, imgsz=self.imgsz, conf=conf, iou=iou, verbose=False)[0]
            sp = r0.speed
            res.preprocess_ms, res.inference_ms, res.postprocess_ms = sp["preprocess"], sp["inference"], sp["postprocess"]
            b = r0.boxes
            boxes = [(*xy, float(c), int(k)) for xy, c, k in
                     zip(b.xyxy.cpu().numpy().tolist(), b.conf.cpu().numpy(), b.cls.cpu().numpy())]
            res.detections = self._make(boxes, (h, w))
        res.total_ms = (time.perf_counter() - t0) * 1e3
        if annotate:
            res.annotated = draw(img, res.detections)
        return res
