import json
import platform
from datetime import datetime
import pandas as pd
import plotly.express as px
import streamlit as st
import config
from modules import edge_runtime as er, metrics, services, ui


def _label(r):
    return r["model"]


def render():
    s = services.settings()
    ui.header("Edge Optimization", "Export, quantise and benchmark models. All numbers are measured on this machine.")
    cap = er.capabilities()
    with st.expander("Detected runtime capabilities", expanded=False):
        ui.status_rows([("ONNX Runtime", cap["onnxruntime"] or "not installed", "green" if cap["onnxruntime"] else "red"),
                        ("Execution providers", ", ".join(cap["ort_providers"]) or "-", "blue"),
                        ("CUDA (PyTorch)", "available" if cap["cuda"] else "not available", "green" if cap["cuda"] else "gray"),
                        ("TensorRT execution provider", "available" if cap["tensorrt_ep"] else "not available", "green" if cap["tensorrt_ep"] else "gray"),
                        ("Ultralytics / onnx", f"{cap['ultralytics'] or 'missing'} / {cap['onnx'] or 'missing'}", "blue"),
                        ("Host", f"{cap['host']} · {cap['cpu_count']} CPU(s)", "gray")])
    files = er.list_model_files()
    st.markdown("#### 1 · Create optimised variants")
    c1, c2, c3 = st.columns(3)
    with c1:
        pts = [f.name for f in files if f.suffix == ".pt"]
        pt = st.selectbox("PyTorch checkpoint", pts or ["(none)"])
        if st.button("Export FP32 ONNX", disabled=not pts):
            try:
                with st.spinner("Exporting..."):
                    out = er.export_onnx(config.MODELS_DIR / pt, s["imgsz"])
                    dst = config.MODELS_DIR / out.name
                    if out.resolve() != dst.resolve():
                        out.replace(dst)
                st.success(f"Exported {dst.name} ({er.size_mb(dst):.1f} MB)")
            except Exception as e:
                st.error(str(e))
    onx = [f.name for f in files if f.suffix == ".onnx"]
    base_onx = [n for n in onx if "int8" not in n and "fp16" not in n]
    with c2:
        src = st.selectbox("FP32 ONNX source", base_onx or ["(none)"])
        if st.button("Create FP16", disabled=not base_onx):
            try:
                out = er.convert_fp16(config.MODELS_DIR / src)
                st.success(f"Created {out.name}. FP16 helps mainly on GPUs; on CPU it can be slower - check the benchmark.")
            except Exception as e:
                st.error(str(e))
    with c3:
        calib = st.text_input("INT8 calibration images", s["sim_dataset_dir"])
        if st.button("Create INT8", disabled=not base_onx):
            try:
                with st.spinner("Quantising..."):
                    out, how = er.quantize_int8(config.MODELS_DIR / src, config.resolve(calib), s["imgsz"])
                st.success(f"Created {out.name} via {how}. Re-check recall on Model Performance - INT8 can reduce accuracy.")
            except Exception as e:
                st.error(str(e))
    ui.note("Pruning is not applied automatically: structured pruning of a YOLO model needs fine-tuning afterwards. "
            "If you prune in your training pipeline, drop the resulting file into <code>models/</code> and benchmark it here.")

    st.markdown("#### 2 · Benchmark")
    files = er.list_model_files()
    if not files:
        ui.empty("No model files in models/", "Add best.pt / best.onnx first.")
        return
    prof = st.selectbox("Edge profile", list(config.EDGE_PROFILES))
    p = config.EDGE_PROFILES[prof]
    ui.note(("<b>SIMULATED edge profile:</b> only caps CPU threads on this host. Latency will NOT match the real device's clock speed, memory bandwidth or accelerator. "
             "Use it to study scaling, and report real-device numbers separately.") if p["simulated"] else
            "<b>Actual measurement</b> on the host running this app.", warn=p["simulated"])
    a, b = st.columns([3, 1])
    chosen = a.multiselect("Models to benchmark", [f.name for f in files], default=[f.name for f in files])
    runs = b.number_input("Timed runs", 10, 500, 50, 10)
    if st.button("Run benchmark", type="primary", disabled=not chosen):
        rows, bar = [], st.progress(0.0)
        for i, name in enumerate(chosen):
            f = config.MODELS_DIR / name
            try:
                rows.append(er.benchmark_onnx(f, s["imgsz"], runs, 5, p["threads"]) if f.suffix == ".onnx" else er.benchmark_pytorch(f, s["imgsz"], max(10, runs // 2)))
            except Exception as e:
                st.error(f"{name}: {e}")
            bar.progress((i + 1) / len(chosen))
        bar.empty()
        metrics.save_json({"timestamp": datetime.now().isoformat(timespec="seconds"), "profile": prof, "simulated": p["simulated"],
                           "host": cap["host"], "cpu_count": cap["cpu_count"], "results": rows}, "edge_benchmarks.json")
    data = metrics.load_json("edge_benchmarks.json")
    if not data or not data["results"]:
        ui.empty("No benchmark yet", "Select models and press Run benchmark.")
        return
    df = pd.DataFrame(data["results"])
    st.markdown(ui.pill("SIMULATED PROFILE" if data["simulated"] else "ACTUAL HOST", "orange" if data["simulated"] else "green") + "  "
                + ui.pill(data["profile"], "gray") + "  " + ui.pill(data["timestamp"], "gray"), unsafe_allow_html=True)
    base_name = st.selectbox("Baseline for optimisation %", df.model.tolist(), index=0)
    base = df[df.model == base_name].iloc[0]
    df["Size reduction %"] = [er.optimisation_pct(base.size_mb, v) for v in df.size_mb]
    df["Latency reduction %"] = [er.optimisation_pct(base.mean_ms, v) for v in df.mean_ms]
    df["< 15 ms"] = df.mean_ms.map(lambda v: "yes" if v < config.TARGET_LATENCY_MS else "no")
    cols = ["model", "format", "provider", "threads", "imgsz", "size_mb", "mean_ms", "p50_ms", "p95_ms", "fps", "rss_mb", "Size reduction %", "Latency reduction %", "< 15 ms"]
    ui.show_df(df[cols].round(2))
    st.caption("Latency = model forward pass on random-noise input (timing is data-independent for these networks). RSS = process memory after loading (shared across models in one process, so treat as indicative).")
    x1, x2, x3 = st.columns(3)
    f1 = px.bar(df, x="model", y="mean_ms", error_y=(df.p95_ms - df.mean_ms), title="Inference latency (mean, bar to p95)", color="format")
    f1.add_hline(y=config.TARGET_LATENCY_MS, line_dash="dash", line_color="#F59E0B", annotation_text="15 ms target")
    ui.show_plot(ui.style_fig(f1), "e_lat", x1)
    ui.show_plot(ui.style_fig(px.bar(df, x="model", y="size_mb", title="Model size (MB)", color="format"), legend=False), "e_size", x2)
    ui.show_plot(ui.style_fig(px.bar(df, x="model", y="fps", title="Throughput (FPS, model only)", color="format"), legend=False), "e_fps", x3)
