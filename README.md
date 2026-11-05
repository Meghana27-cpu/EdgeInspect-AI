# EdgeInspect AI - Real-Time Edge-AI Visual Inspection for Industrial Defect Detection

Streamlit dashboard + YOLO26n (NEU-DET steel surface defects) + ONNX Runtime edge inference + SQLite + simulated MQTT sensors.

## 1. Setup
```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Prepare the NEU-DET dataset
Download NEU-DET (Pascal-VOC XML + images), extract it anywhere, then:
```bash
python scripts/prepare_neudet.py --src /path/to/NEU-DET --dst datasets/neu-det
```
This merges the original train/validation folders and makes a **stratified 70/15/15 train/val/test split** (seed 42),
writes YOLO labels and `data.yaml`. The held-out test split is what you should quote for final numbers.

## 3. Train, validate, export
```bash
python scripts/train.py --data datasets/neu-det/data.yaml --model yolo26n.pt --epochs 100 --imgsz 640 --device 0
```
Outputs: `models/best.pt`, `models/best.onnx`, `results/training_metrics.json` (val), `results/test_metrics.json`, confusion matrix and curves.
If you already have trained weights, just copy them to `models/best.pt` (or `.onnx`) and skip this step.

## 4. Run
```bash
streamlit run app.py
```
Open **System Settings** to confirm the model path, threshold, image size and runtime.

## Pages
| Page | What it does |
|---|---|
| Overview Dashboard | KPIs, 6 charts, recent activity, system status - all from the SQLite records |
| Product Registration | Unique Product IDs, batch, line, category |
| Image Inspection | Upload/camera -> boxes, labels, confidence, severity, latency, PASS/FAIL, PDF |
| Live Inspection | Webcam, uploaded video, or *clearly-labelled simulated conveyor* from dataset images; FPS/latency/alerts |
| Multi-View Inspection | Several views per product; per-image results + product verdict (FAIL if any view fails) |
| Inspection History | Search/filter, detail view, CSV and PDF export |
| Analytics & Reports | Charts from DB, CSV/PDF export |
| Model Performance | Ultralytics metrics + *measured* evaluation of any model file on labelled images (P/R/F1, image-level recall, confusion matrix) |
| Edge Optimization | ONNX export, FP16, INT8 (static QDQ), benchmark with simulated Jetson/RPi thread profiles |
| IoT Monitoring | Simulated belt speed / temperature / vibration / count, MQTT if a broker exists, alerts |
| System Settings | Validated configuration, weights upload, MQTT test |

## Design decisions worth knowing
* **`views/` instead of `pages/`** - Streamlit auto-creates a second navigation from a `pages/` folder, which would duplicate the custom sidebar.
* **Confidence vs severity** - confidence is the model's certainty; severity (Low/Medium/High) is derived from the defect's *size* as a share of the image (thresholds in Settings).
* **Decision rule** - FAIL if at least one detection >= threshold, else PASS. Inspection mode shifts the threshold (High recall <= 0.15, High precision >= 0.50).
* **YOLO26 ONNX output** - the detector handles both the end-to-end/NMS-free `(1,N,6)` layout and legacy `(1,4+nc,A)` layouts.
* **Multi-view** - detections are *not* matched across views; the same defect seen twice is counted twice. The report says so.
* **Simulation labelling** - simulated sensors and simulated edge profiles are labelled in the UI and stored with `simulated=1`. A "simulated Raspberry Pi/Jetson" profile only caps CPU threads; it does not reproduce device speed.
* **No invented numbers** - every KPI/chart reads from the database or from JSON written by real runs; empty states are shown otherwise.

## Tests
```bash
python -m unittest discover -s tests -v
```
Covers: output decoding (both ONNX layouts), letterbox, settings validation, DB + pipeline + filters, multi-view, PDF/CSV, metric matching logic,
MQTT-less sensor simulation + alerts, NEU-DET XML->YOLO conversion, and a stub-based smoke run of every page.

## Meeting the project targets (<15 ms, >98% recall)
Both must be **measured on your hardware / test split**. Use *Model Performance -> Evaluate* for recall and *Edge Optimization -> Benchmark* for latency.
Tips if you miss them: smaller `imgsz` (NEU-DET images are 200x200 - 320 is often enough), INT8 with calibration then re-check recall, `High recall` mode, more epochs/augmentation, YOLO26s.
Report object-level recall (class + IoU>=0.5) and image-level recall separately and state which one the 98% target refers to.
