"""Run with:  python -m unittest discover -s tests -v   (or pytest)"""
import sys, tempfile, unittest, time
from pathlib import Path
import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from modules import database as db, pipeline, metrics, reporting, preprocessing
from modules.detector import decode_output, severity_from_ratio, Detection, InferenceResult, draw
from modules.mqtt_handler import SensorHub


class StubDetector:
    """Returns a fixed detection list so the pipeline can be tested without model weights."""
    names = config.CLASS_NAMES; model_name = "stub.onnx"; backend = "stub"
    def __init__(self, dets): self.dets = dets
    def predict(self, img, conf=0.25, iou=0.45, annotate=True):
        r = InferenceResult(detections=[d for d in self.dets if d.conf >= conf], shape=img.shape[:2],
                            inference_ms=3.0, total_ms=5.0)
        r.annotated = draw(img, r.detections)
        return r


def det(cls=5, conf=0.9, box=(10, 10, 60, 40), shape=(200, 200)):
    ratio = (box[2] - box[0]) * (box[3] - box[1]) / (shape[0] * shape[1])
    return Detection(cls, config.CLASS_NAMES[cls], conf, box, ratio, severity_from_ratio(ratio))


class DecodeTests(unittest.TestCase):
    def test_end_to_end_layout(self):
        out = np.zeros((1, 300, 6), np.float32)
        out[0, 0] = [10, 20, 110, 120, 0.8, 3]; out[0, 1] = [0, 0, 5, 5, 0.1, 1]
        r = decode_output(out, 6, 0.25, 0.45)
        self.assertEqual(len(r), 1); self.assertEqual(r[0][5], 3)

    def test_legacy_layout_with_nms(self):
        a = np.zeros((1, 10, 5), np.float32)       # (4+nc, anchors)
        for i, (cx, conf) in enumerate([(100, .9), (102, .8), (300, .7)]):
            a[0, :4, i] = [cx, 100, 50, 50]; a[0, 4 + 2, i] = conf   # class 2
        r = decode_output(a, 6, 0.25, 0.45)
        self.assertEqual(len(r), 2)                 # overlapping pair merged by NMS
        self.assertTrue(all(x[5] == 2 for x in r))

    def test_bad_shape_raises(self):
        with self.assertRaises(ValueError):
            decode_output(np.zeros((1, 7, 9), np.float32), 6, .25, .45)

    def test_severity_independent_of_confidence(self):
        self.assertEqual(severity_from_ratio(0.01), "Low")
        self.assertEqual(severity_from_ratio(0.05), "Medium")
        self.assertEqual(severity_from_ratio(0.20), "High")


class PreprocessTests(unittest.TestCase):
    def test_letterbox_roundtrip(self):
        img = np.full((200, 300, 3), 128, np.uint8)
        x, r, (px, py) = preprocessing.to_tensor(img, 640)
        self.assertEqual(x.shape, (1, 3, 640, 640)); self.assertAlmostEqual(r, 640 / 300, places=4)
        self.assertEqual(px, 0); self.assertGreater(py, 0)

    def test_grayscale_decode(self):
        ok, buf = cv2.imencode(".png", np.zeros((50, 50), np.uint8))
        self.assertEqual(preprocessing.decode_image(buf.tobytes()).shape, (50, 50, 3))
        with self.assertRaises(ValueError):
            preprocessing.decode_image(b"not an image")


class SettingsTests(unittest.TestCase):
    def test_defaults_valid(self):
        s, e = config.validate_settings({}); self.assertEqual(e, [])
    def test_invalid_values_reported(self):
        s, e = config.validate_settings({"conf_threshold": 5, "imgsz": 100, "model_path": "x.txt", "edge_runtime": "tpu"})
        self.assertGreaterEqual(len(e), 4)
    def test_modes(self):
        s = dict(config.DEFAULT_SETTINGS)
        s["inspection_mode"] = "High recall"; self.assertEqual(config.effective_conf(s), 0.15)
        s["inspection_mode"] = "High precision"; self.assertEqual(config.effective_conf(s), 0.5)


class DbPipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dbp = Path(self.tmp.name) / "t.db"
        db.init_db(self.dbp)
        self.s = dict(config.DEFAULT_SETTINGS)
        self.img = np.full((200, 200, 3), 90, np.uint8)

    def tearDown(self): self.tmp.cleanup()

    def test_duplicate_product_rejected(self):
        self.assertTrue(db.add_product("P1", "Strip", "Steel plate", "B1", "L1", db_path=self.dbp))
        self.assertFalse(db.add_product("P1", "Other", "Steel plate", "B2", "L2", db_path=self.dbp))

    def test_pass_and_fail_flow(self):
        db.add_product("P1", "Strip", "Steel plate", "B1", "L1", db_path=self.dbp)
        r1 = pipeline.run_inspection(StubDetector([]), self.img, self.s, "P1", db_path=self.dbp)
        r2 = pipeline.run_inspection(StubDetector([det(), det(0, 0.6, (50, 50, 150, 150))]), self.img, self.s, "P1", db_path=self.dbp)
        self.assertEqual((r1["status"], r2["status"]), ("PASS", "FAIL"))
        k = db.kpis(self.dbp)
        self.assertEqual((k["total"], k["passed"], k["failed"], k["defects"]), (2, 1, 1, 2))
        self.assertTrue((config.BASE_DIR / r2["annotated_path"]).exists())
        self.assertEqual(len(db.get_detections(r2["inspection_id"], self.dbp)), 2)
        self.assertEqual(len(db.list_alerts(db_path=self.dbp)), 1)

    def test_threshold_controls_decision(self):
        r = pipeline.run_inspection(StubDetector([det(conf=0.2)]), self.img, self.s, persist=False)
        self.assertEqual(r["status"], "PASS")        # 0.2 < default 0.25
        self.s["inspection_mode"] = "High recall"
        r = pipeline.run_inspection(StubDetector([det(conf=0.2)]), self.img, self.s, persist=False)
        self.assertEqual(r["status"], "FAIL")

    def test_filters_and_multiview_and_pdf(self):
        db.add_product("P2", "Coil", "Steel coil", "B", "L", db_path=self.dbp)
        sid = pipeline.new_session_id()
        a = pipeline.run_inspection(StubDetector([det()]), self.img, self.s, "P2", "multiview", "Front", sid, db_path=self.dbp)
        b = pipeline.run_inspection(StubDetector([]), self.img, self.s, "P2", "multiview", "Back", sid, db_path=self.dbp)
        v = db.session_views(sid, self.dbp)
        self.assertEqual(sorted(v.view_name), ["Back", "Front"])
        self.assertEqual(len(db.query_inspections(product_id="P2", status="FAIL", db_path=self.dbp)), 1)
        self.assertEqual(len(db.query_inspections(defect="scratches", db_path=self.dbp)), 1)
        self.assertEqual(len(db.query_inspections(defect="crazing", db_path=self.dbp)), 0)
        self.assertEqual(len(db.query_inspections(date_from="2999-01-01", db_path=self.dbp)), 0)
        pdf = reporting.build_pdf([a["inspection_id"], b["inspection_id"]], Path(self.tmp.name) / "r.pdf", db_path=self.dbp)
        self.assertTrue(pdf.read_bytes().startswith(b"%PDF"))
        self.assertIn(b"inspection_type", reporting.csv_bytes(db.query_inspections(db_path=self.dbp)))


class MetricsTests(unittest.TestCase):
    def test_evaluate_detector_counts(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d); (d / "images").mkdir(); (d / "labels").mkdir()
            img = np.full((200, 200, 3), 100, np.uint8)
            for n in ("a", "b"):
                cv2.imwrite(str(d / "images" / f"{n}.png", ), img)
                (d / "labels" / f"{n}.txt").write_text("5 0.2 0.15 0.25 0.15\n")   # box 10,10,60,40
            # detector finds the right box with the right class in every image
            res = metrics.evaluate_detector(StubDetector([det()]), d / "images", d / "labels", conf=.25)
            self.assertEqual(res["overall"]["recall"], 1.0); self.assertEqual(res["image_level_recall"], 1.0)
            # detector predicts wrong class -> counted as FP+FN, recall 0
            res = metrics.evaluate_detector(StubDetector([det(cls=0)]), d / "images", d / "labels", conf=.25)
            self.assertEqual(res["overall"]["recall"], 0.0); self.assertEqual(res["image_level_recall"], 1.0)
            # detector finds nothing -> recall 0
            res = metrics.evaluate_detector(StubDetector([]), d / "images", d / "labels", conf=.25)
            self.assertEqual(res["overall"]["fn"], 2)


class HubTests(unittest.TestCase):
    def test_local_simulation_and_alert(self):
        with tempfile.TemporaryDirectory() as d:
            dbp = Path(d) / "h.db"; db.init_db(dbp)
            s = dict(config.DEFAULT_SETTINGS); s["mqtt_enabled"] = False
            hub = SensorHub(s, db_path=dbp, tick=0.05); hub.start()
            time.sleep(0.4); snap = hub.snapshot()
            self.assertEqual(snap["mode"], "Local simulation"); self.assertTrue(snap["running"])
            self.assertGreater(len(snap["hist"]["temperature"]), 3)
            hub.inject_fault("temperature", 5); time.sleep(0.6)
            self.assertIn("temperature", hub.snapshot()["abnormal"])
            hub.stop()
            self.assertFalse(hub.running)
            self.assertGreaterEqual(len(db.list_alerts(source="iot", db_path=dbp)), 1)
            self.assertGreater(len(db.recent_sensor_readings(db_path=dbp)), 0)


def tearDownModule():
    """Remove images the pipeline wrote into uploads/ and results/ during tests."""
    for d, pre in ((config.UPLOAD_DIR, "orig_"), (config.RESULTS_DIR, "ann_")):
        for f in d.glob(pre + "*.jpg"):
            f.unlink()


if __name__ == "__main__":
    unittest.main()
