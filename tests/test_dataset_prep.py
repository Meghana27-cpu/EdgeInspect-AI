import sys, tempfile, unittest
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from scripts import prepare_neudet as prep
from modules.preprocessing import validate_yolo_dataset

XML = """<annotation><filename>{n}.jpg</filename><size><width>200</width><height>200</height><depth>1</depth></size>
<object><name>{c}</name><bndbox><xmin>20</xmin><ymin>40</ymin><xmax>120</xmax><ymax>100</ymax></bndbox></object></annotation>"""


class PrepTests(unittest.TestCase):
    def test_conversion_split_and_validation(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d); src = d / "src"
            (src / "images").mkdir(parents=True); (src / "ann").mkdir()
            for c in ["crazing", "rolled-in_scale", "pitted_surface"]:
                for i in range(1, 21):
                    n = f"{c}_{i}"
                    cv2.imwrite(str(src / "images" / f"{n}.jpg"), np.full((200, 200), 120, np.uint8))
                    (src / "ann" / f"{n}.xml").write_text(XML.format(n=n, c=c))
            (src / "ann" / "bad.xml").write_text("<annotation><object><name>unknown</name></object></annotation>")
            rep = prep.convert(src, d / "out", (0.7, 0.15, 0.15), 1)
            self.assertEqual((rep["images"], rep["train"], rep["val"], rep["test"]), (60, 42, 9, 9))
            self.assertEqual(len(rep["skipped"]), 1)
            lab = next((d / "out/labels/train").glob("crazing_*.txt")).read_text().split()
            self.assertEqual(lab[0], "0")
            self.assertAlmostEqual(float(lab[1]), 0.35); self.assertAlmostEqual(float(lab[2]), 0.35)
            self.assertAlmostEqual(float(lab[3]), 0.5); self.assertAlmostEqual(float(lab[4]), 0.3)
            self.assertTrue(validate_yolo_dataset(d / "out", config.CLASS_NAMES)["ok"])
            self.assertIn("rolled-in_scale", (d / "out/data.yaml").read_text())
            self.assertEqual(prep.class_id("Rolled-in Scale"), 4)

if __name__ == "__main__":
    unittest.main()
