import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import config
from modules import edge_runtime, metrics, services, ui
from modules.detector import Detector
from modules.preprocessing import validate_yolo_dataset


def _target_message(rec: float, n: int):
    if rec >= config.TARGET_RECALL:
        ui.note(f"<b>Recall target met on this evaluation set:</b> {rec:.1%} ≥ {config.TARGET_RECALL:.0%} ({n} images). "
                "Confirm on a held-out test split that was not used to pick the checkpoint or threshold.")
    else:
        ui.note(f"<b>Recall target NOT met:</b> measured {rec:.1%} vs target {config.TARGET_RECALL:.0%}. "
                "Ideas: lower the confidence threshold (High-recall mode), train longer / larger image size, try yolo26s, "
                "add augmentation, and inspect the per-class table - crazing and rolled-in scale are typically the hardest NEU-DET classes.", warn=True)


def render():
    s = services.settings()
    ui.header("Model Performance", "Training metrics and measured evaluation of the deployed model")
    t1, t2 = st.tabs(["Training / validation metrics", "Evaluate deployed model"])
    with t1:
        m = metrics.load_json("training_metrics.json")
        if not m:
            ui.empty("No training metrics found", "Run  python scripts/train.py  (writes results/training_metrics.json), or evaluate the deployed model in the next tab.")
        else:
            o = m["overall"]
            ui.kpi_row([("Precision", f"{o['precision']:.3f}", m.get("split", "val") + " split", "blue"), ("Recall", f"{o['recall']:.3f}", f"target > {config.TARGET_RECALL}", "green" if o["recall"] >= config.TARGET_RECALL else "orange"),
                        ("F1", f"{o['f1']:.3f}", "from P and R", "cyan"), ("mAP50", f"{o['map50']:.3f}", "IoU 0.50", "blue"), ("mAP50-95", f"{o['map50_95']:.3f}", "COCO-style", "blue")])
            st.caption(f"Model: {m.get('weights')} · {m.get('timestamp')} · produced by Ultralytics validation")
            _target_message(o["recall"], m.get("images", 0))
            pc = pd.DataFrame(m["per_class"]).T.reset_index().rename(columns={"index": "Class"})
            ui.show_df(pc)
            for f, cap in (("confusion_matrix_normalized.png", "Confusion matrix (normalised)"), ("results.png", "Training curves")):
                if (config.RESULTS_DIR / f).exists():
                    st.image(str(config.RESULTS_DIR / f), caption=cap)
    with t2:
        files = edge_runtime.list_model_files()
        if not files:
            ui.empty("No model files in models/")
            return
        names = [f.name for f in files]
        c = st.columns([1.3, 1.5, 1, 1, 1])
        model = c[0].selectbox("Model file", names, index=names.index(config.resolve(s["model_path"]).name) if config.resolve(s["model_path"]).name in names else 0)
        root = c[1].text_input("Dataset root (YOLO format)", "datasets/neu-det")
        split = c[2].selectbox("Split", ["val", "test", "train"])
        conf = c[3].slider("Confidence", 0.01, 0.9, float(config.effective_conf(s)), 0.01)
        mx = c[4].number_input("Max images (0 = all)", 0, 100000, 0)
        if st.button("Run evaluation", type="primary"):
            rootp = config.resolve(root)
            ip, lp = rootp / "images" / split, rootp / "labels" / split
            if not ip.is_dir():
                st.error(f"Folder not found: {ip}")
            else:
                det = Detector(config.MODELS_DIR / model, s["imgsz"], "auto", s["ort_threads"])
                if not det.loaded:
                    st.error(det.error)
                else:
                    bar = st.progress(0.0, "Evaluating...")
                    res = metrics.evaluate_detector(det, ip, lp, conf, s["iou_threshold"], max_images=mx or None, progress=bar.progress)
                    res["dataset"], res["split"] = str(rootp), split
                    metrics.save_json(res, "deployed_eval.json")
                    bar.empty()
        r = metrics.load_json("deployed_eval.json")
        if not r:
            ui.empty("No evaluation yet", "Metrics shown here come from actually running the selected model on labelled images.")
            return
        o = r["overall"]
        st.caption(f"Last run: {r['model']} · {r['backend']} · {r['images']} images ({r.get('split')}) · conf {r['conf']} · IoU match {r['iou_match']} · {r['timestamp']}")
        ui.kpi_row([("Object recall", f"{o['recall']:.3f}", "TP / (TP + FN)", "green" if o["recall"] >= config.TARGET_RECALL else "orange"),
                    ("Object precision", f"{o['precision']:.3f}", f"{o['fp']} false positives", "blue"), ("F1", f"{o['f1']:.3f}", "", "cyan"),
                    ("Class-agnostic recall", f"{r['class_agnostic']['recall']:.3f}" if "class_agnostic" in r else "-", "defect found at IoU ≥ 0.5, class ignored", "cyan"),
                    ("Image-level recall", f"{r['image_level_recall']:.3f}" if r["image_level_recall"] is not None else "–", "defective images with ≥1 detection", "green" if (r["image_level_recall"] or 0) >= config.TARGET_RECALL else "orange"),
                    ("Mean latency", f"{r['latency_ms']['mean']:.1f} ms", f"p95 {r['latency_ms']['p95']:.1f} ms", "green" if r["latency_ms"]["mean"] < config.TARGET_LATENCY_MS else "orange")])
        _target_message(o["recall"], r["images"])
        ui.note("Object recall requires the correct class <i>and</i> IoU ≥ 0.5; image-level recall only asks whether a defective image was flagged at all. "
                "Both are shown because the project target (&gt;98% recall) can be read either way - state which one you use in the report. "
                "mAP is reported by Ultralytics validation (first tab), not recomputed here.")
        pc = pd.DataFrame(r["per_class"]).T
        pc.index = [config.PRETTY.get(i, i) for i in pc.index]
        ui.show_df(pc.round(3).reset_index().rename(columns={"index": "Class"}))
        cm = np.array(r["confusion_matrix"])
        fig = px.imshow(cm, x=r["labels"], y=r["labels"], text_auto=True, color_continuous_scale="Blues", aspect="auto",
                        labels=dict(x="Predicted", y="Ground truth", color="Count"), title="Confusion matrix (rows = ground truth)")
        ui.show_plot(ui.style_fig(fig, 480, False), "mp_cm")
    with st.expander("Dataset validation"):
        root = st.text_input("YOLO dataset root", "datasets/neu-det", key="valroot")
        if st.button("Validate dataset"):
            rep = validate_yolo_dataset(config.resolve(root), config.CLASS_NAMES)
            (st.success if rep["ok"] else st.error)("Dataset OK" if rep["ok"] else "; ".join(rep["problems"]))
            st.json(rep["splits"])
