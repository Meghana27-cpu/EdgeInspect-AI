"""SQLite persistence layer (products, inspections, detections, views, sensors, alerts)."""
from __future__ import annotations
import sqlite3
from contextlib import contextmanager
from datetime import datetime
import pandas as pd
import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS products(
  product_id TEXT PRIMARY KEY, name TEXT NOT NULL, category TEXT, batch TEXT,
  production_line TEXT, description TEXT, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS inspections(
  id INTEGER PRIMARY KEY AUTOINCREMENT, product_id TEXT, session_id TEXT,
  inspection_type TEXT NOT NULL DEFAULT 'image', view_name TEXT, timestamp TEXT NOT NULL,
  original_path TEXT, annotated_path TEXT, status TEXT NOT NULL, num_defects INTEGER NOT NULL DEFAULT 0,
  max_confidence REAL, inference_ms REAL, total_ms REAL, model_name TEXT, backend TEXT, threshold REAL,
  FOREIGN KEY(product_id) REFERENCES products(product_id));
CREATE TABLE IF NOT EXISTS detections(
  id INTEGER PRIMARY KEY AUTOINCREMENT, inspection_id INTEGER NOT NULL, defect_class TEXT NOT NULL,
  confidence REAL NOT NULL, severity TEXT, area_ratio REAL, x1 REAL, y1 REAL, x2 REAL, y2 REAL,
  FOREIGN KEY(inspection_id) REFERENCES inspections(id) ON DELETE CASCADE);
CREATE TABLE IF NOT EXISTS product_views(
  id INTEGER PRIMARY KEY AUTOINCREMENT, product_id TEXT NOT NULL, session_id TEXT NOT NULL,
  view_name TEXT NOT NULL, inspection_id INTEGER, image_path TEXT, created_at TEXT NOT NULL,
  FOREIGN KEY(product_id) REFERENCES products(product_id));
CREATE TABLE IF NOT EXISTS sensor_readings(
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, sensor TEXT NOT NULL, value REAL,
  unit TEXT, simulated INTEGER NOT NULL DEFAULT 1, abnormal INTEGER NOT NULL DEFAULT 0);
CREATE TABLE IF NOT EXISTS alerts(
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, source TEXT NOT NULL, level TEXT NOT NULL,
  message TEXT NOT NULL, product_id TEXT);
CREATE INDEX IF NOT EXISTS ix_insp_ts ON inspections(timestamp);
CREATE INDEX IF NOT EXISTS ix_insp_prod ON inspections(product_id);
CREATE INDEX IF NOT EXISTS ix_det_insp ON detections(inspection_id);
"""


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@contextmanager
def conn(db_path=None):
    c = sqlite3.connect(str(db_path or config.DB_PATH), timeout=10)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA foreign_keys=ON")
    try:
        yield c
        c.commit()
    finally:
        c.close()


def init_db(db_path=None) -> None:
    config.ensure_dirs()
    with conn(db_path) as c:
        c.executescript(SCHEMA)
        c.execute("PRAGMA journal_mode=WAL")


def check_db(db_path=None) -> bool:
    try:
        with conn(db_path) as c:
            c.execute("SELECT 1 FROM inspections LIMIT 1")
        return True
    except sqlite3.Error:
        return False


def _df(sql, params=(), db_path=None) -> pd.DataFrame:
    with conn(db_path) as c:
        return pd.read_sql_query(sql, c, params=params)


# ---------------------------------------------------------------- products
def add_product(product_id, name, category, batch, line, description="", db_path=None) -> bool:
    """Return False if the product ID already exists."""
    try:
        with conn(db_path) as c:
            c.execute("INSERT INTO products VALUES(?,?,?,?,?,?,?)",
                      (product_id.strip(), name.strip(), category, batch.strip(), line.strip(), description.strip(), now()))
        return True
    except sqlite3.IntegrityError:
        return False


def get_product(product_id, db_path=None) -> dict | None:
    with conn(db_path) as c:
        r = c.execute("SELECT * FROM products WHERE product_id=?", (product_id,)).fetchone()
    return dict(r) if r else None


def list_products(db_path=None) -> pd.DataFrame:
    return _df("""SELECT p.*, COUNT(i.id) AS inspections,
                  SUM(CASE WHEN i.status='FAIL' THEN 1 ELSE 0 END) AS failed
                  FROM products p LEFT JOIN inspections i ON i.product_id=p.product_id
                  GROUP BY p.product_id ORDER BY p.created_at DESC""", db_path=db_path)


# ------------------------------------------------------------- inspections
def add_inspection(rec: dict, detections: list[dict], db_path=None) -> int:
    cols = ["product_id", "session_id", "inspection_type", "view_name", "timestamp", "original_path",
            "annotated_path", "status", "num_defects", "max_confidence", "inference_ms", "total_ms",
            "model_name", "backend", "threshold"]
    rec = {**rec}
    rec.setdefault("timestamp", now())
    with conn(db_path) as c:
        cur = c.execute(f"INSERT INTO inspections({','.join(cols)}) VALUES({','.join('?' * len(cols))})",
                        [rec.get(k) for k in cols])
        iid = cur.lastrowid
        c.executemany("INSERT INTO detections(inspection_id,defect_class,confidence,severity,area_ratio,x1,y1,x2,y2)"
                      " VALUES(?,?,?,?,?,?,?,?,?)",
                      [(iid, d["defect_class"], d["confidence"], d["severity"], d["area_ratio"], *d["box"]) for d in detections])
        if rec.get("session_id") and rec.get("product_id") and rec.get("view_name"):
            c.execute("INSERT INTO product_views(product_id,session_id,view_name,inspection_id,image_path,created_at)"
                      " VALUES(?,?,?,?,?,?)", (rec["product_id"], rec["session_id"], rec["view_name"], iid,
                                               rec.get("original_path"), rec["timestamp"]))
    return iid


def query_inspections(product_id="", date_from=None, date_to=None, defect="All", status="All",
                      itype="All", limit=5000, db_path=None) -> pd.DataFrame:
    where, p = [], []
    if product_id:
        where.append("i.product_id LIKE ?"); p.append(f"%{product_id}%")
    if date_from:
        where.append("date(i.timestamp)>=date(?)"); p.append(str(date_from))
    if date_to:
        where.append("date(i.timestamp)<=date(?)"); p.append(str(date_to))
    if status in ("PASS", "FAIL"):
        where.append("i.status=?"); p.append(status)
    if itype != "All":
        where.append("i.inspection_type=?"); p.append(itype)
    if defect != "All":
        where.append("EXISTS(SELECT 1 FROM detections d WHERE d.inspection_id=i.id AND d.defect_class=?)"); p.append(defect)
    sql = "SELECT i.* FROM inspections i" + (" WHERE " + " AND ".join(where) if where else "") + \
          " ORDER BY i.timestamp DESC, i.id DESC LIMIT ?"
    return _df(sql, (*p, limit), db_path)


def get_inspection(iid, db_path=None) -> dict | None:
    with conn(db_path) as c:
        r = c.execute("SELECT * FROM inspections WHERE id=?", (int(iid),)).fetchone()
    return dict(r) if r else None


def get_detections(inspection_id, db_path=None) -> pd.DataFrame:
    return _df("SELECT * FROM detections WHERE inspection_id=? ORDER BY confidence DESC", (int(inspection_id),), db_path)


def all_detections(db_path=None) -> pd.DataFrame:
    return _df("""SELECT d.*, i.timestamp, i.product_id, i.status, i.inspection_type
                  FROM detections d JOIN inspections i ON i.id=d.inspection_id""", db_path=db_path)


def session_views(session_id, db_path=None) -> pd.DataFrame:
    return _df("""SELECT i.* FROM product_views v JOIN inspections i ON i.id=v.inspection_id
                  WHERE v.session_id=? ORDER BY v.id""", (session_id,), db_path)


def kpis(db_path=None) -> dict:
    with conn(db_path) as c:
        r = c.execute("""SELECT COUNT(*) n, COALESCE(SUM(status='PASS'),0) p, COALESCE(SUM(status='FAIL'),0) f,
                         COALESCE(SUM(num_defects),0) d, AVG(inference_ms) inf, AVG(total_ms) tot FROM inspections""").fetchone()
        last = c.execute("SELECT AVG(total_ms) FROM (SELECT total_ms FROM inspections ORDER BY id DESC LIMIT 20)").fetchone()[0]
    n = r["n"]
    return {"total": n, "passed": r["p"], "failed": r["f"], "defects": r["d"],
            "fail_rate": (r["f"] / n * 100) if n else 0.0,
            "avg_inference_ms": r["inf"], "avg_total_ms": r["tot"],
            "fps": (1000.0 / last) if last else None}


# -------------------------------------------------------- sensors & alerts
def add_sensor_readings(rows: list[tuple], db_path=None) -> None:
    """rows: (sensor, value, unit, simulated, abnormal)"""
    t = now()
    with conn(db_path) as c:
        c.executemany("INSERT INTO sensor_readings(ts,sensor,value,unit,simulated,abnormal) VALUES(?,?,?,?,?,?)",
                      [(t, *r) for r in rows])


def recent_sensor_readings(limit=500, db_path=None) -> pd.DataFrame:
    return _df("SELECT * FROM sensor_readings ORDER BY id DESC LIMIT ?", (limit,), db_path)


def add_alert(source, level, message, product_id=None, db_path=None) -> None:
    with conn(db_path) as c:
        c.execute("INSERT INTO alerts(ts,source,level,message,product_id) VALUES(?,?,?,?,?)",
                  (now(), source, level, message, product_id))


def list_alerts(limit=100, source="All", db_path=None) -> pd.DataFrame:
    if source != "All":
        return _df("SELECT * FROM alerts WHERE source=? ORDER BY id DESC LIMIT ?", (source, limit), db_path)
    return _df("SELECT * FROM alerts ORDER BY id DESC LIMIT ?", (limit,), db_path)
