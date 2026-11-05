"""Executes every page's render() against a stub Streamlit/Plotly. Only runs when the real packages
are NOT installed (it is a stand-in for a real UI run, not a replacement - run the app to verify layout)."""
import importlib.util, sys, tempfile, types, unittest
from pathlib import Path
from unittest.mock import MagicMock
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
REAL = importlib.util.find_spec("streamlit") is not None and importlib.util.find_spec("plotly") is not None


class SS(dict):
    __getattr__ = dict.get
    def __setattr__(self, k, v): self[k] = v


def make_stub(st=None):
    if st is None:
        st = MagicMock(name="st")
        st.session_state = SS()
    first = lambda opts, *a, **k: (list(opts)[k.get("index", 0) or 0] if len(list(opts)) else None)
    st.columns.side_effect = lambda n, **k: [st for _ in range(n if isinstance(n, int) else len(n))]
    st.tabs.side_effect = lambda names: [st for _ in names]
    for n in ('expander', 'form', 'spinner', 'container', 'progress'):
        getattr(st, n).return_value = st
    st.selectbox.side_effect = lambda label, opts, *a, **k: first(opts, **k)
    st.radio.side_effect = lambda label, opts, *a, **k: first(opts, **k)
    st.multiselect.side_effect = lambda label, opts, *a, default=None, **k: default or []
    st.text_input.side_effect = lambda label, value="", *a, **k: value
    st.text_area.side_effect = lambda *a, **k: ""
    st.number_input.side_effect = lambda label, mn=0, mx=0, value=0, *a, **k: value
    st.slider.side_effect = lambda label, mn, mx, value=None, *a, **k: value if value is not None else mn
    st.checkbox.side_effect = lambda label, value=False, *a, **k: value
    st.toggle.side_effect = lambda label, value=False, *a, **k: value
    st.date_input.side_effect = lambda *a, **k: []
    st.file_uploader.side_effect = lambda *a, **k: [] if k.get("accept_multiple_files") else None
    st.button.return_value = False
    st.form_submit_button.return_value = False
    st.camera_input.return_value = None
    st.cache_resource = lambda *a, **k: (a[0] if a and callable(a[0]) else (lambda f: f))
    st.fragment = lambda *a, **k: (lambda f: f)
    st.empty.return_value = st
    return st



@unittest.skipIf(REAL, "real streamlit/plotly installed - run the app instead")
class ViewSmoke(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.st = make_stub()
        sys.modules["streamlit"] = cls.st
        for m in ("plotly", "plotly.express", "plotly.graph_objects"):
            sys.modules[m] = MagicMock(name=m)
        import config
        from modules import database as db, pipeline
        from modules.detector import Detection, InferenceResult, draw, severity_from_ratio
        cls.tmp = tempfile.TemporaryDirectory()
        config.DB_PATH = Path(cls.tmp.name) / "s.db"
        db.init_db(config.DB_PATH)
        import modules.database as dbm
        dbm.conn.__wrapped__  # ensure contextmanager present
        db.add_product("P1", "Strip", "Steel plate", "B1", "L1")
        class Stub:
            loaded = True; names = config.CLASS_NAMES; model_name = "stub.onnx"; backend = "stub"; imgsz = 640; error = ""
            def predict(self, img, conf=.25, iou=.45, annotate=True):
                b = (10, 10, 60, 40)
                d = Detection(5, "scratches", .9, b, 1500 / 40000, severity_from_ratio(1500 / 40000))
                r = InferenceResult([d], None, 1, 3, 1, 5, img.shape[:2]); r.annotated = draw(img, [d]); return r
        cls.stub = Stub()
        img = np.full((200, 200, 3), 90, np.uint8)
        for _ in range(3):
            pipeline.run_inspection(cls.stub, img, config.DEFAULT_SETTINGS, "P1")
        from modules import services
        services.detector = lambda *a, **k: cls.stub
        s = dict(config.DEFAULT_SETTINGS); s["mqtt_enabled"] = False
        cls.st.session_state["settings"] = s

    def setUp(self):
        keep = self.st.session_state
        self.st.reset_mock(return_value=True, side_effect=True)
        self.st.session_state = keep
        make_stub(self.st)
        for k in ("img_results", "mv", "live_on"):
            keep.pop(k, None)

    def _run(self, name):
        import importlib
        importlib.import_module(f"views.{name}").render()

    def test_all_views_render(self):
        for name in ("dashboard", "product_registration", "image_inspection", "live_inspection", "multi_view",
                     "inspection_history", "analytics", "model_performance", "edge_optimization", "iot_monitoring", "settings"):
            with self.subTest(view=name):
                self._run(name)

    def test_action_paths(self):
        """Press the action buttons with real files and confirm DB rows are written."""
        import cv2, importlib, config
        from modules import database as db
        st = self.st
        ok, buf = cv2.imencode(".png", np.full((200, 200, 3), 80, np.uint8))
        class F:  # minimal UploadedFile
            name = "x.png"
            def getvalue(self): return buf.tobytes()
        before = db.kpis()["total"]
        # image inspection: two uploaded files
        st.button.side_effect = lambda label, *a, **k: label == "Run inspection"
        st.file_uploader.side_effect = lambda *a, **k: [F(), F()] if k.get("accept_multiple_files") else None
        importlib.import_module("views.image_inspection").render()
        self.assertEqual(db.kpis()["total"], before + 2)
        self.assertEqual(len(st.session_state["img_results"]), 2)
        # multi-view: front + back
        st.multiselect.side_effect = lambda label, opts, *a, default=None, **k: ["Front", "Back"]
        st.button.side_effect = lambda label, *a, **k: label == "Run multi-view inspection"
        st.file_uploader.side_effect = lambda *a, **k: F()
        importlib.import_module("views.multi_view").render()
        self.assertEqual(db.kpis()["total"], before + 4)
        self.assertEqual(len(st.session_state["mv"]["results"]), 2)
        # live simulation over a folder of 3 images
        d = Path(self.tmp.name) / "sim"; d.mkdir()
        for i in range(3):
            cv2.imwrite(str(d / f"{i}.png"), np.full((200, 200, 3), 60 + i, np.uint8))
        st.session_state["settings"]["sim_dataset_dir"] = str(d)
        st.radio.side_effect = lambda label, opts, *a, **k: "Dataset conveyor simulation"
        st.checkbox.side_effect = lambda label, value=False, *a, **k: False if label == "Loop source" else value
        st.slider.side_effect = lambda label, mn, mx, value=None, *a, **k: 30
        st.button.side_effect = lambda label, *a, **k: label == "Start inspection"
        importlib.import_module("views.live_inspection").render()
        self.assertFalse(st.session_state["live_on"])           # source finished cleanly
        self.assertGreaterEqual(len(db.query_inspections(itype="live")), 1)


def tearDownModule():
    """Remove images the pipeline wrote into uploads/ and results/ during tests."""
    for d, pre in ((__import__('config').UPLOAD_DIR, "orig_"), (__import__('config').RESULTS_DIR, "ann_")):
        for f in d.glob(pre + "*.jpg"):
            f.unlink()


if __name__ == "__main__":
    unittest.main()
