import json
import time
import os
import sqlite3
from dataclasses import dataclass, asdict, field
from typing import Optional
from collections import deque

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


@dataclass
class SessionRecord:
    timestamp: float
    duration_sec: float
    avg_bpm: float
    avg_resp: float
    avg_hrv_sdnn: float
    avg_stress: float
    avg_sqi: float
    arrhythmia_detected: bool
    notes: str = ""


@dataclass
class AIInterpretation:
    summary: str        = ""
    risk_flags: list    = field(default_factory=list)
    advice: list        = field(default_factory=list)
    disclaimer: str     = "Not a medical diagnosis. Consult a healthcare professional."
    raw_response: str   = ""


class SessionHistory:
    def __init__(self, db_path: str = "rppg_sessions.db"):
        self.db_path = db_path
        self._init_db()
        self.current_session_start = time.time()
        self.bpm_samples    = deque(maxlen=1000)
        self.resp_samples   = deque(maxlen=1000)
        self.hrv_samples    = deque(maxlen=1000)
        self.stress_samples = deque(maxlen=1000)
        self.sqi_samples    = deque(maxlen=1000)
        self.arrhythmia_events = []

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL, duration_sec REAL,
            avg_bpm REAL, avg_resp REAL, avg_hrv_sdnn REAL,
            avg_stress REAL, avg_sqi REAL,
            arrhythmia_detected INTEGER, notes TEXT
        )''')
        c.execute('''CREATE TABLE IF NOT EXISTS readings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER, timestamp REAL,
            bpm REAL, resp REAL, hrv_sdnn REAL,
            stress REAL, sqi REAL
        )''')
        conn.commit()
        conn.close()

    def ingest(self, bpm, resp, hrv_sdnn, stress, sqi, arrhythmia=False):
        if bpm > 0:    self.bpm_samples.append(bpm)
        if resp > 0:   self.resp_samples.append(resp)
        if hrv_sdnn > 0: self.hrv_samples.append(hrv_sdnn)
        if stress > 0: self.stress_samples.append(stress)
        if sqi > 0:    self.sqi_samples.append(sqi)
        if arrhythmia: self.arrhythmia_events.append(time.time())

    def save_session(self, notes: str = "") -> Optional[int]:
        if not self.bpm_samples:
            return None
        duration = time.time() - self.current_session_start
        rec = SessionRecord(
            timestamp          = self.current_session_start,
            duration_sec       = duration,
            avg_bpm            = float(sum(self.bpm_samples) / len(self.bpm_samples)),
            avg_resp           = float(sum(self.resp_samples) / len(self.resp_samples)) if self.resp_samples else 0,
            avg_hrv_sdnn       = float(sum(self.hrv_samples) / len(self.hrv_samples)) if self.hrv_samples else 0,
            avg_stress         = float(sum(self.stress_samples) / len(self.stress_samples)) if self.stress_samples else 0,
            avg_sqi            = float(sum(self.sqi_samples) / len(self.sqi_samples)) if self.sqi_samples else 0,
            arrhythmia_detected = len(self.arrhythmia_events) > 0,
            notes              = notes
        )
        conn = sqlite3.connect(self.db_path)
        c    = conn.cursor()
        c.execute('''INSERT INTO sessions
            (timestamp, duration_sec, avg_bpm, avg_resp, avg_hrv_sdnn,
             avg_stress, avg_sqi, arrhythmia_detected, notes)
            VALUES (?,?,?,?,?,?,?,?,?)''',
            (rec.timestamp, rec.duration_sec, rec.avg_bpm, rec.avg_resp,
             rec.avg_hrv_sdnn, rec.avg_stress, rec.avg_sqi,
             int(rec.arrhythmia_detected), rec.notes))
        session_id = c.lastrowid
        conn.commit()
        conn.close()
        self._reset_session()
        return session_id

    def _reset_session(self):
        self.current_session_start = time.time()
        self.bpm_samples.clear()
        self.resp_samples.clear()
        self.hrv_samples.clear()
        self.stress_samples.clear()
        self.sqi_samples.clear()
        self.arrhythmia_events.clear()

    def get_history(self, limit: int = 50) -> list:
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''SELECT timestamp, duration_sec, avg_bpm, avg_resp,
                     avg_hrv_sdnn, avg_stress, avg_sqi, arrhythmia_detected, notes
                     FROM sessions ORDER BY timestamp DESC LIMIT ?''', (limit,))
        rows = c.fetchall()
        conn.close()
        return rows

    def get_trend_summary(self) -> dict:
        rows = self.get_history(limit=20)
        if not rows:
            return {}
        bpms    = [r[2] for r in rows if r[2] > 0]
        stresses = [r[5] for r in rows if r[5] > 0]
        hrvs    = [r[4] for r in rows if r[4] > 0]
        return {
            "session_count":   len(rows),
            "avg_bpm_trend":   round(sum(bpms) / len(bpms), 1)    if bpms    else 0,
            "avg_stress_trend": round(sum(stresses) / len(stresses), 1) if stresses else 0,
            "avg_hrv_trend":   round(sum(hrvs)  / len(hrvs),  1)   if hrvs    else 0,
            "arrhythmia_sessions": sum(1 for r in rows if r[7])
        }


class WellnessGuidanceEngine:
    def __init__(self):
        self.last_guidance_time = 0
        self.cooldown_sec = 8.0
        self.guidance_queue = deque(maxlen=5)
        self._last_frame_state = {}

    def check_webcam_quality(self, fps: float, motion_score: float,
                              sqi: float, face_detected: bool,
                              mean_brightness: float) -> list:
        tips = []
        if not face_detected:
            tips.append(("FACE", "No face detected — center your face in frame"))
        elif sqi < 20:
            if mean_brightness < 60:
                tips.append(("LIGHT", "Too dark — face more light sources"))
            elif mean_brightness > 220:
                tips.append(("LIGHT", "Overexposed — reduce backlight or move away from window"))
            if motion_score > 2.8:  # FIX2: synced with cfg.MOTION_ENTER_THRESHOLD (was 4 – gaslighting users)
                tips.append(("MOTION", "Too much movement — stay still for better reading"))
            else:
                tips.append(("DIST", "Poor signal — try moving 40-70cm from camera"))
        elif sqi < 40:
            if motion_score > 1.8:  # FIX2: synced with cfg.MOTION_EXIT_THRESHOLD (was 2.5)
                tips.append(("MOTION", "Minor motion detected — stay still"))
            if mean_brightness < 80:
                tips.append(("LIGHT", "Improve lighting for better accuracy"))
        return tips

    def realtime_face_guidance(self, motion_score: float, sqi: float,
                                face_detected: bool, mean_brightness: float,
                                fused_bpm: float) -> Optional[str]:
        now = time.time()
        if now - self.last_guidance_time < self.cooldown_sec:
            return self._last_guidance if hasattr(self, '_last_guidance') else None

        msg = None
        if not face_detected:
            msg = "Position your face in the center of the frame"
        elif motion_score > 2.8:  # FIX2: synced with cfg.MOTION_ENTER_THRESHOLD (was 5.0)
            msg = "Please stay still for an accurate reading"
        elif mean_brightness < 55:
            msg = "Move to better lighting or turn on more lights"
        elif mean_brightness > 215:
            msg = "High backlight detected — face the light source"
        elif sqi < 25 and fused_bpm > 0:
            msg = "Signal weak — try: direct light on face, less movement"
        elif sqi > 55 and fused_bpm > 0:
            msg = "Signal quality good — hold still"

        if msg:
            self.last_guidance_time = now
            self._last_guidance = msg
        return msg

    def sqi_explanation(self, sqi_breakdown: dict, motion_score: float,
                         mean_brightness: float) -> list:
        explanations = []
        if not sqi_breakdown:
            return explanations

        snr  = sqi_breakdown.get("snr", 0)
        reg  = sqi_breakdown.get("regularity", 0)
        var  = sqi_breakdown.get("variance", 0)
        m_pen = sqi_breakdown.get("motion_penalty", 0)
        l_pen = sqi_breakdown.get("lighting_penalty", 0)

        if m_pen > 20:
            explanations.append(f"Motion artifact ({motion_score:.1f}px/frame) reducing signal quality")
        elif m_pen > 8:
            explanations.append(f"Minor movement detected — try to stay still")

        if l_pen > 20:
            if mean_brightness > 210:
                explanations.append(f"Overexposure (brightness={mean_brightness:.0f}) — face away from window")
            elif mean_brightness < 55:
                explanations.append(f"Underexposure (brightness={mean_brightness:.0f}) — add front lighting")
        elif l_pen > 8:
            if mean_brightness > 190:
                explanations.append("Slight overexposure — reduce backlight")
            elif mean_brightness < 70:
                explanations.append("Low ambient light — improve room lighting")
        if snr < 25:
            explanations.append(f"Weak pulse signal (SNR={snr:.0f}%) — skin coverage or lighting issue")
        elif snr < 45:
            explanations.append(f"Moderate SNR ({snr:.0f}%) — signal collecting")
        if reg < 25:
            explanations.append(f"Irregular signal pattern (reg={reg:.0f}%) — possible motion or noise")
        if var < 15:
            explanations.append(f"Signal too flat — check ROI skin coverage")

        overall = sqi_breakdown.get("overall", 0)
        if not explanations and overall >= 55:
            explanations.append(f"Signal quality good (SQI={overall:.0f}%)")

        return explanations[:3] 


class MedicalAILayer:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key   = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.cache     = {}
        self.last_call = 0
        self.cooldown  = 30.0

    def _build_prompt(self, vitals: dict, baseline_dev: dict, trend: dict) -> str:
        return f"""You are a research-grade physiological signal analysis assistant.
Analyze the following real-time rPPG (remote photoplethysmography) measurements and provide:
1. A concise 2-sentence clinical summary
2. Up to 3 specific risk flags (if any — be conservative, not alarmist)
3. Up to 3 actionable wellness suggestions

Current measurements:
- Heart Rate: {vitals.get('bpm', 0):.0f} BPM
- Respiration Rate: {vitals.get('resp_rate', 0):.1f} breaths/min
- HRV SDNN: {vitals.get('hrv_sdnn', 0):.1f} ms
- HRV RMSSD: {vitals.get('hrv_rmssd', 0):.1f} ms
- LF/HF Ratio: {vitals.get('lf_hf_ratio', 0):.2f}
- Stress Index: {vitals.get('stress_index', 0):.1f}/100
- Signal Quality (SQI): {vitals.get('sqi', 0):.0f}%
- Arrhythmia flag: {vitals.get('arrhythmia_flag', False)}
- Arrhythmia detail: {vitals.get('arrhythmia_detail', 'none')}

Personal baseline deviation: {json.dumps(baseline_dev)}
Session trend (last 20 sessions): {json.dumps(trend)}

Respond ONLY in this JSON format:
{{
  "summary": "...",
  "risk_flags": ["...", "..."],
  "advice": ["...", "...", "..."]
}}

Important: This is a non-clinical research tool. Be conservative with risk flags.
Do not diagnose. Flag only when values are clearly outside normal ranges."""

    def interpret(self, vitals: dict, baseline_dev: dict = None,
                  trend: dict = None) -> AIInterpretation:
        result = AIInterpretation()

        if not HAS_REQUESTS or not self.api_key:
            result.summary = self._rule_based_summary(vitals)
            result.advice  = self._rule_based_advice(vitals)
            return result

        now = time.time()
        if now - self.last_call < self.cooldown:
            return self._cached_or_default(vitals)

        try:
            prompt = self._build_prompt(
                vitals,
                baseline_dev or {},
                trend or {}
            )
            resp = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key":         self.api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type":      "application/json"
                },
                json={
                    "model":      "claude-sonnet-4-20250514",
                    "max_tokens": 600,
                    "messages":   [{"role": "user", "content": prompt}]
                },
                timeout=8
            )
            if resp.status_code == 200:
                text = resp.json()["content"][0]["text"]
                parsed = json.loads(text)
                result.summary     = parsed.get("summary", "")
                result.risk_flags  = parsed.get("risk_flags", [])
                result.advice      = parsed.get("advice", [])
                result.raw_response = text
                self.cache["last"] = result
                self.last_call     = now
            else:
                result = self._cached_or_default(vitals)

        except Exception as e:
            result = self._cached_or_default(vitals)

        return result

    def _cached_or_default(self, vitals: dict) -> AIInterpretation:
        if "last" in self.cache:
            return self.cache["last"]
        result = AIInterpretation()
        result.summary = self._rule_based_summary(vitals)
        result.advice  = self._rule_based_advice(vitals)
        return result

    def _rule_based_summary(self, v: dict) -> str:
        bpm     = v.get("bpm", 0)
        stress  = v.get("stress_index", 0)
        hrv     = v.get("hrv_sdnn", 0)

        if bpm <= 0:
            return "Insufficient data for analysis."

        bpm_label  = "normal" if 60 <= bpm <= 100 else ("elevated" if bpm > 100 else "low")
        stress_str = f"Stress index is {'elevated' if stress > 50 else 'moderate' if stress > 30 else 'low'} at {stress:.0f}/100."
        return f"Heart rate is {bpm_label} at {bpm:.0f} BPM. {stress_str}"

    def _rule_based_advice(self, v: dict) -> list:
        advice = []
        bpm    = v.get("bpm", 0)
        stress = v.get("stress_index", 0)
        resp   = v.get("resp_rate", 0)
        hrv    = v.get("hrv_sdnn", 0)

        if bpm > 100:
            advice.append("Elevated HR: consider rest or relaxation techniques")
        if stress > 50:
            advice.append("High stress detected: try 4-7-8 breathing exercise")
        if resp > 20:
            advice.append("Fast breathing: slow down — aim for 12-16 breaths/min")
        if hrv < 20 and hrv > 0:
            advice.append("Low HRV: consider more sleep and reduced stimulants")
        if not advice:
            advice.append("Vitals appear within normal range — keep it up")
        return advice[:3]
