"""Edge optimisation: ONNX export, FP16/INT8 conversion, and *measured* benchmarks."""
from __future__ import annotations
import os
import platform
import shutil
import time
from pathlib import Path
import numpy as np
import config
from modules.preprocessing import list_images, to_tensor


def rss_mb() -> float | None:
    try:
        import psutil
        return psutil.Process().memory_info().rss / 2**20
    except ImportError:
        try:
            import resource
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # peak RSS on Linux (KB)
        except Exception:
            return None


def capabilities() -> dict:
    cap = {"host": f"{platform.system()} {platform.machine()}", "cpu_count": os.cpu_count(), "python": platform.python_version()}
    def ver(mod):
        try:
            return __import__(mod).__version__
        except Exception:
            return None
    for m in ("onnxruntime", "onnx", "ultralytics", "torch", "psutil", "paho"):
        cap[m] = ver(m) if m != "paho" else (ver("paho.mqtt") or ("installed" if _has("paho.mqtt.client") else None))
    try:
        import onnxruntime as ort
        cap["ort_providers"] = ort.get_available_providers()
    except ImportError:
        cap["ort_providers"] = []
    try:
        import torch
        cap["cuda"] = bool(torch.cuda.is_available())
    except Exception:
        cap["cuda"] = False
    cap["tensorrt_ep"] = "TensorrtExecutionProvider" in cap["ort_providers"]
    return cap


def _has(mod) -> bool:
    try:
        __import__(mod); return True
    except Exception:
        return False


def size_mb(path) -> float:
    return Path(path).stat().st_size / 2**20


def export_onnx(pt_path, imgsz=640, half=False, dynamic=False, simplify=True) -> Path:
    """Export a .pt checkpoint to ONNX through Ultralytics (must be installed)."""
    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise RuntimeError("ultralytics is not installed - `pip install ultralytics`") from e
    out = YOLO(str(config.resolve(pt_path))).export(format="onnx", imgsz=imgsz, half=half,
                                                    dynamic=dynamic, simplify=simplify, opset=17)
    return Path(out)


def convert_fp16(onnx_path, out_path=None) -> Path:
    try:
        import onnx
        from onnxconverter_common import float16
    except ImportError as e:
        raise RuntimeError("FP16 conversion needs `pip install onnx onnxconverter-common`") from e
    out = Path(out_path or Path(onnx_path).with_name(Path(onnx_path).stem + "_fp16.onnx"))
    m = float16.convert_float_to_float16(onnx.load(str(onnx_path)), keep_io_types=False)
    onnx.save(m, str(out))
    return out


def quantize_int8(onnx_path, calib_dir=None, imgsz=640, n_calib=64, out_path=None) -> tuple[Path, str]:
    """INT8 quantisation with ONNX Runtime. Static (calibrated) if calibration images exist, else dynamic.
    Returns (path, method). Always re-check accuracy afterwards: INT8 can lower recall."""
    try:
        import cv2
        from onnxruntime.quantization import (CalibrationDataReader, QuantFormat, QuantType,
                                              quantize_dynamic, quantize_static)
        import onnxruntime as ort
    except ImportError as e:
        raise RuntimeError("INT8 quantisation needs onnxruntime and `pip install onnx`") from e
    src = Path(onnx_path)
    out = Path(out_path or src.with_name(src.stem.replace("_fp16", "") + "_int8.onnx"))
    imgs = list_images(calib_dir)[:n_calib] if calib_dir else []
    if imgs:
        name = ort.InferenceSession(str(src), providers=["CPUExecutionProvider"]).get_inputs()[0].name

        class Reader(CalibrationDataReader):
            def __init__(self):
                self.it = iter(imgs)
            def get_next(self):
                p = next(self.it, None)
                if p is None:
                    return None
                return {name: to_tensor(cv2.imread(str(p)), imgsz)[0]}
        quantize_static(str(src), str(out), Reader(), quant_format=QuantFormat.QDQ,
                        activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8, per_channel=True)
        return out, f"static QDQ ({len(imgs)} calibration images)"
    quantize_dynamic(str(src), str(out), weight_type=QuantType.QInt8)
    return out, "dynamic (no calibration images found; conv layers may not be accelerated)"


def _percentiles(t: list[float]) -> dict:
    a = np.array(t)
    return {"mean_ms": float(a.mean()), "p50_ms": float(np.percentile(a, 50)),
            "p95_ms": float(np.percentile(a, 95)), "min_ms": float(a.min()), "std_ms": float(a.std()),
            "fps": float(1000 / a.mean())}


def _bench_image(imgsz, image=None):
    if image is not None:
        return image
    rng = np.random.default_rng(0)
    return (rng.random((imgsz, imgsz, 3)) * 255).astype(np.uint8)


def benchmark_onnx(path, imgsz=640, runs=50, warmup=5, threads=0, providers=None) -> dict:
    """Pure model-forward latency of an ONNX file on this machine."""
    import onnxruntime as ort
    before = rss_mb()
    so = ort.SessionOptions()
    if threads > 0:
        so.intra_op_num_threads = threads
    sess = ort.InferenceSession(str(path), so, providers=providers or ["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    h = inp.shape[2] if isinstance(inp.shape[2], int) else imgsz
    x, _, _ = to_tensor(_bench_image(h), h, fp16="float16" in inp.type)
    for _ in range(warmup):
        sess.run(None, {inp.name: x})
    t = []
    for _ in range(runs):
        s = time.perf_counter()
        sess.run(None, {inp.name: x})
        t.append((time.perf_counter() - s) * 1000)
    after = rss_mb()
    return {"model": Path(path).name, "format": "ONNX", "size_mb": size_mb(path), "imgsz": h, "runs": runs,
            "threads": threads or "default", "provider": sess.get_providers()[0], **_percentiles(t),
            "rss_mb": after, "rss_delta_mb": (after - before) if after and before else None,
            "input_data": "random noise (latency only)"}


def benchmark_pytorch(path, imgsz=640, runs=30, warmup=3) -> dict:
    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise RuntimeError("ultralytics is not installed") from e
    m = YOLO(str(path))
    img = _bench_image(imgsz)
    before = rss_mb()
    for _ in range(warmup):
        m.predict(img, imgsz=imgsz, verbose=False)
    t = [m.predict(img, imgsz=imgsz, verbose=False)[0].speed["inference"] for _ in range(runs)]
    after = rss_mb()
    return {"model": Path(path).name, "format": "PyTorch", "size_mb": size_mb(path), "imgsz": imgsz, "runs": runs,
            "threads": "default", "provider": "PyTorch", **_percentiles(t), "rss_mb": after,
            "rss_delta_mb": (after - before) if after and before else None, "input_data": "random noise (latency only)"}


def optimisation_pct(base: float, new: float) -> float:
    return (base - new) / base * 100 if base else 0.0


def list_model_files() -> list[Path]:
    return sorted(p for p in config.MODELS_DIR.glob("*") if p.suffix.lower() in (".pt", ".onnx"))
