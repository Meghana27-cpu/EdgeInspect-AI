"""Simulated industrial IoT sensors with optional MQTT transport (paho-mqtt).

If no broker is reachable the hub runs in *local simulation* mode. Every reading is simulated
(there is no physical hardware) and is labelled as such in the UI and database.
"""
from __future__ import annotations
import json
import random
import socket
import threading
import time
from collections import deque
import config
from modules import database as db

SENSORS = {  # name: (unit, nominal, noise)
    "belt_speed": ("m/s", 1.5, 0.05),
    "temperature": ("°C", 45.0, 0.6),
    "vibration": ("mm/s", 2.0, 0.15),
}


class SensorHub:
    def __init__(self, settings: dict, db_path=None, tick: float = 1.0):
        self.s = settings
        self.db_path = db_path
        self.tick = tick
        self.hist = {k: deque(maxlen=300) for k in (*SENSORS, "production_count")}
        self.values = {k: v[1] for k, v in SENSORS.items()}
        self.production = 0
        self.status = "STOPPED"
        self.mode = "Local simulation"
        self.mqtt_error = ""
        self.faults: dict[str, float] = {}      # sensor -> expiry time
        self._last_alert: dict[str, float] = {}
        self._client = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._ticks = 0
        self.alerts_raised = 0

    # --------------------------------------------------------------- control
    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self):
        if self.running:
            return
        self._stop.clear()
        self._connect()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="sensor-hub")
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)
        if self._client:
            try:
                self._client.loop_stop(); self._client.disconnect()
            except Exception:
                pass
            self._client = None
        self.status = "STOPPED"

    def inject_fault(self, sensor: str, seconds: float = 12.0):
        if sensor in SENSORS:
            self.faults[sensor] = time.time() + seconds

    # ------------------------------------------------------------------ MQTT
    def _connect(self):
        self.mode, self.mqtt_error = "Local simulation", ""
        if not self.s.get("mqtt_enabled"):
            self.mqtt_error = "MQTT disabled in settings"
            return
        try:
            import paho.mqtt.client as mqtt
        except ImportError:
            self.mqtt_error = "paho-mqtt not installed"
            return
        try:
            socket.create_connection((self.s["mqtt_host"], int(self.s["mqtt_port"])), timeout=1.5).close()
            try:
                c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
            except AttributeError:       # paho 1.x
                c = mqtt.Client()
            c.on_message = self._on_message
            c.connect(self.s["mqtt_host"], int(self.s["mqtt_port"]), keepalive=30)
            c.subscribe(f"{self.s['mqtt_prefix']}/sensors/#")
            c.loop_start()
            self._client = c
            self.mode = f"MQTT broker {self.s['mqtt_host']}:{self.s['mqtt_port']}"
        except Exception as e:
            self.mqtt_error = f"Broker unreachable ({type(e).__name__}: {e}) - using local simulation"

    def _on_message(self, client, userdata, msg):
        try:
            d = json.loads(msg.payload)
            self._ingest(d["sensor"], float(d["value"]))
        except Exception:
            pass

    def publish_event(self, suffix: str, payload: dict):
        if self._client is not None:
            try:
                self._client.publish(f"{self.s['mqtt_prefix']}/{suffix}", json.dumps(payload))
            except Exception:
                pass

    # ------------------------------------------------------------ simulation
    def _generate(self) -> dict:
        now = time.time()
        out = {}
        for k, (unit, nominal, noise) in SENSORS.items():
            if self.faults.get(k, 0) > now:
                v = {"belt_speed": 0.1, "temperature": self.s["temp_max"] + 8, "vibration": self.s["vibration_max"] + 2.5}[k]
                v += random.gauss(0, noise)
            else:
                v = self.values[k] + 0.3 * (nominal - self.values[k]) + random.gauss(0, noise)
            out[k] = v
        return out

    def _loop(self):
        self.status = "RUNNING"
        while not self._stop.is_set():
            for sensor, v in self._generate().items():
                if self._client is not None:
                    self.publish_event(f"sensors/{sensor}", {"sensor": sensor, "value": v, "simulated": True})
                else:
                    self._ingest(sensor, v)
            with self._lock:
                self.production += max(0, int(round(self.values["belt_speed"] * 2 + random.random())))
                self.hist["production_count"].append((time.time(), self.production))
                bad = self.abnormal()
                self.status = "FAULT" if bad else "RUNNING"
            self._ticks += 1
            if self._ticks % 5 == 0:
                self._persist(bad)
            self._stop.wait(self.tick)

    def _ingest(self, sensor, v):
        if sensor not in SENSORS:
            return
        with self._lock:
            self.values[sensor] = v
            self.hist[sensor].append((time.time(), v))

    def abnormal(self) -> list[str]:
        v, s = self.values, self.s
        bad = []
        if not s["belt_speed_min"] <= v["belt_speed"] <= s["belt_speed_max"]:
            bad.append("belt_speed")
        if v["temperature"] > s["temp_max"]:
            bad.append("temperature")
        if v["vibration"] > s["vibration_max"]:
            bad.append("vibration")
        return bad

    def _persist(self, bad):
        try:
            rows = [(k, self.values[k], SENSORS[k][0], 1, int(k in bad)) for k in SENSORS]
            rows.append(("production_count", self.production, "units", 1, 0))
            db.add_sensor_readings(rows, self.db_path)
            for k in bad:
                if time.time() - self._last_alert.get(k, 0) > 30:
                    self._last_alert[k] = time.time()
                    self.alerts_raised += 1
                    db.add_alert("iot", "critical", f"[SIMULATED] {k} out of range: {self.values[k]:.2f} {SENSORS[k][0]}",
                                 db_path=self.db_path)
        except Exception:
            pass

    def snapshot(self) -> dict:
        with self._lock:
            return {"values": dict(self.values), "production": self.production, "status": self.status,
                    "mode": self.mode, "abnormal": self.abnormal(), "running": self.running,
                    "hist": {k: list(v) for k, v in self.hist.items()}}
