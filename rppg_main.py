import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rppg_config import cfg
from rppg_config import cfg as _cfg
from rppg_core import (MultiROIFusionEngine, MultiROIFusionEngineV2,
                        ROI_CONFIGS, BUFFER_SIZE,
                        SQI_HARD_GATE, MOTION_ENTER_THRESHOLD, MOTION_EXIT_THRESHOLD)
from rppg_vitals import VitalsEngine, VitalsResult
from rppg_ai import SessionHistory, WellnessGuidanceEngine, MedicalAILayer
from typing import Optional


PALETTE = {
    "bg":          (10,  12,  22),
    "panel":       (15,  18,  35),
    "green":       (0,   220, 100),
    "green_dim":   (0,   120, 60),
    "cyan":        (0,   210, 220),
    "yellow":      (30,  220, 220),
    "orange":      (30,  160, 255),
    "red":         (50,  50,  240),
    "red_bright":  (80,  80,  255),
    "white":       (220, 220, 230),
    "gray":        (100, 100, 120),
    "gray_dim":    (50,  50,  70),
    "purple":      (200, 100, 220),
    "teal":        (180, 220, 100),
}

MIN_FRAMES = _cfg.MIN_FRAMES


def draw_panel(img, x1, y1, x2, y2, alpha=0.75):
    sub = img[y1:y2, x1:x2]
    bg  = np.full_like(sub, PALETTE["panel"])
    cv2.addWeighted(bg, alpha, sub, 1 - alpha, 0, sub)
    img[y1:y2, x1:x2] = sub
    cv2.rectangle(img, (x1, y1), (x2, y2), PALETTE["gray_dim"], 1)


def put_text(img, text, x, y, scale=0.5, color=None, thickness=1, font=cv2.FONT_HERSHEY_DUPLEX):
    color = color or PALETTE["white"]
    cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def draw_signal_graph(canvas, signal, h, w, color=None, label="", bpm=0.0, sqi=0.0,
                       sqi_breakdown=None, session_confidence=0.0):
    color = color or PALETTE["green"]
    canvas[:] = np.array(PALETTE["bg"], dtype=np.uint8)

    for i in range(1, 6):
        y = int(i * h / 6)
        cv2.line(canvas, (0, y), (w, y), PALETTE["gray_dim"], 1)
    for i in range(1, 12):
        x = int(i * w / 12)
        cv2.line(canvas, (x, 0), (x, h), PALETTE["gray_dim"], 1)

    if len(signal) < 4:
        put_text(canvas, label, 10, 22, 0.5, PALETTE["gray"])
        return

    sig = np.array(signal, dtype=float)
    rng = sig.max() - sig.min()
    if rng < 1e-8: rng = 1.0
    xs  = np.linspace(0, w - 1, len(sig)).astype(int)
    ys  = ((sig - sig.min()) / rng)
    ys  = ((1.0 - ys) * h * 0.86 + h * 0.07).astype(int)
    pts = np.stack([xs, ys], axis=1).reshape(-1, 1, 2)
    cv2.polylines(canvas, [pts], False, color, 2, cv2.LINE_AA)

    put_text(canvas, label, 10, 20, 0.45, PALETTE["gray"])

    if bpm > 0:
        bpm_c = PALETTE["green"] if 50 <= bpm <= 100 else PALETTE["cyan"]
        put_text(canvas, f"BPM: {bpm:.0f}", w - 280, 20, 0.65, bpm_c, 2)


    bar_x, bar_y, bar_w, bar_h = w - 130, 28, 110, 9
    if sqi > 0:
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), PALETTE["gray_dim"], -1)
        filled = int(bar_w * min(sqi, 100) / 100.0)
        c_sqi  = PALETTE["green"] if sqi >= 65 else PALETTE["yellow"] if sqi >= 40 else PALETTE["red"]
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + filled, bar_y + bar_h), c_sqi, -1)
        put_text(canvas, f"SQI:{sqi:.0f}%", bar_x, bar_y - 2, 0.42, c_sqi)


    if session_confidence > 0:
        sc_c = PALETTE["green"] if session_confidence >= 70 else \
               PALETTE["yellow"] if session_confidence >= 45 else PALETTE["red"]
        put_text(canvas, f"Session:{session_confidence:.0f}%", w - 130, bar_y + 20, 0.38, sc_c)


def draw_spectrum_graph(canvas, freqs, power, peak_bpm, h, w):
    canvas[:] = np.array(PALETTE["bg"], dtype=np.uint8)
    if freqs is None or len(freqs) == 0:
        put_text(canvas, "FFT — collecting data...", 12, h // 2, 0.5, PALETTE["gray"])
        return

    mask = (freqs >= 0.67) & (freqs <= 3.4)
    if not np.any(mask):
        return

    f_s = freqs[mask]
    p_s = power[mask]
    p_max = p_s.max()
    if p_max < 1e-9: return
    p_n = p_s / p_max
    n   = len(f_s)

    for i, (f, p) in enumerate(zip(f_s, p_n)):
        x1 = int(i * w / n)
        x2 = max(x1 + 1, int((i + 1) * w / n) - 1)
        bh = int(p * (h - 25))
        in_range = (0.70 <= f <= 3.33)
        color    = (20, int(60 + 160 * p), int(40 + 80 * p)) if in_range else PALETTE["gray_dim"]
        cv2.rectangle(canvas, (x1, h - 18 - bh), (x2, h - 18), color, -1)

    for bpm_l in [45, 60, 75, 90, 110, 130, 150, 180]:
        f_hz = bpm_l / 60.0
        if 0.67 <= f_hz <= 3.4:
            xp = int((f_hz - 0.67) / (3.4 - 0.67) * w)
            cv2.line(canvas, (xp, 0), (xp, h - 18), PALETTE["gray_dim"], 1)
            put_text(canvas, str(bpm_l), max(0, xp - 10), h - 4, 0.35, PALETTE["gray"])

    put_text(canvas, f"FFT Spectrum  |  Peak: {peak_bpm:.0f} BPM", 10, 14, 0.42, PALETTE["gray"])


def draw_vitals_panel(canvas, vitals: VitalsResult, baseline_dev: dict, h: int, w: int):
    draw_panel(canvas, 0, 0, w, h)


    sqi_val = getattr(vitals, "sqi", 0.0)
    bpm_color = (PALETTE["green"]  if sqi_val >= 55 else
                 PALETTE["yellow"] if sqi_val >= 35 else
                 PALETTE["orange"] if sqi_val >= 18 else
                 PALETTE["red_bright"])
    bpm_label = PALETTE["red_bright"] if vitals.stress_index > 60 else PALETTE["yellow"]

    rows = [

        ("BPM",      f"{vitals.bpm:.0f}" if vitals.bpm > 0 else "--",
                                             "bpm",    bpm_color),
        ("RESP",     f"{vitals.resp_rate:.1f}",   "br/min",  PALETTE["cyan"]),
        ("HRV SDNN", f"{vitals.hrv_sdnn:.1f}",    "ms",      PALETTE["purple"]),
        ("HRV RMSSD", f"{vitals.hrv_rmssd:.1f}",  "ms",      PALETTE["purple"]),
        ("pNN50",    f"{vitals.hrv_pnn50:.1f}",    "%",       PALETTE["yellow"]),
        ("LF/HF",    f"{vitals.lf_hf_ratio:.2f}",  "",       PALETTE["orange"]),
        ("STRESS",   f"{vitals.stress_index:.0f}", "/100",    bpm_label),
    ]

    col_w = w // 3
    for i, (label, val, unit, color) in enumerate(rows):
        col  = i % 3
        row  = i // 3
        cx   = col * col_w + 8
        cy   = row * (h // 3) + 28
        put_text(canvas, label, cx, cy - 14, 0.38, PALETTE["gray"])
        put_text(canvas, val,   cx, cy,      0.72, color, 2)
        if unit:
            put_text(canvas, unit, cx + len(val) * 10 + 4, cy, 0.38, PALETTE["gray"])

    bpm_z = baseline_dev.get("bpm_z", None)
    if bpm_z is not None:
        z_color = PALETTE["red_bright"] if abs(bpm_z) > 2 else PALETTE["green"]
        put_text(canvas, f"BPM deviation: {bpm_z:+.1f}z", w - 190, h - 10, 0.4, z_color)


def draw_roi_overlay(frame, roi_signals: dict, motion_rejected: bool):
    roi_colors = {
        "forehead":    (0, 220, 100),
        "cheek_left":  (0, 180, 220),
        "cheek_right": (220, 180, 0),
    }

    for roi_name, roi in roi_signals.items():
        if not roi.valid or roi.pts is None:
            continue
        color = roi_colors.get(roi_name, (200, 200, 200))
        if motion_rejected:
            color = (80, 80, 255)

        overlay_poly = frame.copy()
        cv2.fillPoly(overlay_poly, [roi.pts], color)
        cv2.addWeighted(overlay_poly, 0.15, frame, 0.85, 0, frame)
        cv2.polylines(frame, [roi.pts], True, color, 2, cv2.LINE_AA)

        x1, y1, x2, y2 = roi.bbox

        sqi_c = PALETTE["green"] if roi.sqi >= 65 else PALETTE["yellow"] if roi.sqi >= 40 else PALETTE["red"]

        skin_cov = getattr(roi, 'skin_coverage', 1.0)
        put_text(frame, f"{roi_name[:3].upper()} SQI:{roi.sqi:.0f}% W:{roi.dynamic_weight:.1f} sk:{skin_cov:.0%}",
                 x1, max(12, y1 - 5), 0.38, sqi_c)


def draw_arrhythmia_warning(frame, vitals: VitalsResult, h: int, w: int):
    if not vitals.arrhythmia_flag:
        return
    t  = time.time()
    if int(t * 2) % 2 == 0:
        cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 180), 3)
    bx, by = w // 2 - 220, 85
    draw_panel(frame, bx, by, bx + 440, by + 55)
    put_text(frame, "! IRREGULAR RHYTHM — SUSTAINED DETECTION",
             bx + 10, by + 18, 0.52, PALETTE["red_bright"], 2)
    put_text(frame, vitals.arrhythmia_detail[:60],
             bx + 10, by + 34, 0.38, PALETTE["orange"])
    put_text(frame, "Multi-window validated | Not a diagnosis",
             bx + 10, by + 48, 0.33, PALETTE["gray"])


def draw_guidance(frame, tips: list, guidance_msg: Optional[str],
                  sqi_explanations: list, h: int, w: int,
                  exposure_drift: float = 0.0):
    y = h - 70
    for icon, tip in tips[:3]:
        icon_color = {
            "LIGHT":  PALETTE["yellow"],
            "MOTION": PALETTE["orange"],
            "FACE":   PALETTE["cyan"],
            "DIST":   PALETTE["gray"],
            "EXPO":   PALETTE["teal"],
        }.get(icon, PALETTE["white"])
        put_text(frame, f"[{icon}] {tip}", 10, y, 0.42, icon_color)
        y += 18


    if exposure_drift > 35:
        put_text(frame, f"[EXPO] Camera auto-exposure drift: {exposure_drift:.1f} — normalizing",
                 10, y, 0.42, PALETTE["teal"])
        y += 18

    if guidance_msg:
        put_text(frame, f">> {guidance_msg}", 10, h - 10, 0.48, PALETTE["green"])

    if sqi_explanations:
        ex_text = "WHY: " + " | ".join(sqi_explanations[:2])
        put_text(frame, ex_text, 10, h - 28, 0.38, PALETTE["gray"])


def draw_ai_panel(canvas, ai_result, h: int, w: int):
    draw_panel(canvas, 0, 0, w, h)
    if not ai_result:
        put_text(canvas, "AI Layer: press [A] to analyze", 10, h // 2, 0.45, PALETTE["gray"])
        return

    put_text(canvas, "AI INTERPRETATION", 10, 18, 0.45, PALETTE["cyan"])
    words = ai_result.summary.split()
    line, lines = "", []
    for w_tok in words:
        if len(line) + len(w_tok) + 1 > 90:
            lines.append(line)
            line = w_tok
        else:
            line = (line + " " + w_tok).strip()
    if line: lines.append(line)
    for i, ln in enumerate(lines[:3]):
        put_text(canvas, ln, 10, 36 + i * 16, 0.38, PALETTE["white"])

    y = 36 + len(lines[:3]) * 16 + 8
    for flag in ai_result.risk_flags[:2]:
        put_text(canvas, f"! {flag[:80]}", 10, y, 0.36, PALETTE["orange"])
        y += 14
    for adv in ai_result.advice[:2]:
        put_text(canvas, f"+ {adv[:80]}", 10, y, 0.36, PALETTE["green_dim"])
        y += 14

    put_text(canvas, ai_result.disclaimer[:80], 10, h - 8, 0.32, PALETTE["gray_dim"])


def draw_history_panel(canvas, history_rows: list, trend: dict, h: int, w: int):
    draw_panel(canvas, 0, 0, w, h)
    put_text(canvas, "SESSION HISTORY", 10, 18, 0.45, PALETTE["cyan"])

    headers = ["TIME", "BPM", "RESP", "HRV", "STRESS", "SQI", "ARR"]
    col_xs  = [10, 80, 130, 175, 220, 275, 320]
    for i, (hdr, cx) in enumerate(zip(headers, col_xs)):
        put_text(canvas, hdr, cx, 34, 0.32, PALETTE["gray"])

    for i, row in enumerate(history_rows[-4:]):
        y = 50 + i * 14
        put_text(canvas, row["time"], col_xs[0], y, 0.32, PALETTE["gray"])
        put_text(canvas, f"{row['bpm']:.0f}", col_xs[1], y, 0.35, PALETTE["green"])
        put_text(canvas, f"{row['resp']:.0f}", col_xs[2], y, 0.35, PALETTE["cyan"])
        put_text(canvas, f"{row['hrv']:.0f}", col_xs[3], y, 0.35, PALETTE["purple"])
        put_text(canvas, f"{row['stress']:.0f}", col_xs[4], y, 0.35, PALETTE["red_bright"] if row['stress']>60 else PALETTE["yellow"])
        put_text(canvas, f"{row['sqi']:.0f}%", col_xs[5], y, 0.35, PALETTE["gray"])
        if row["arr"]:
            put_text(canvas, "YES", col_xs[6], y, 0.35, PALETTE["red_bright"])

    if trend:
        t_text = f"Trend: BPM {trend['bpm_trend']} | Stress {trend['stress_trend']} | Stability: {trend['stability']}"
        put_text(canvas, t_text, 10, h - 8, 0.32, PALETTE["gray"])


def main():

    cap = None
    for idx in [1, 2]:
        temp_cap = cv2.VideoCapture(idx)
        if temp_cap.isOpened():
            cap = temp_cap
            print(f"[INFO] Camera found at index {idx}")
            break
        temp_cap.release()

    if cap is None:
        print("[ERROR] No camera found at index 0, 1, or 2.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  Camera: {W}x{H}")

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    )

    fusion_engine  = MultiROIFusionEngineV2(enable_logging=False)
    vitals_engine  = VitalsEngine()
    session_hist   = SessionHistory()
    guidance_eng   = WellnessGuidanceEngine()
    ai_layer       = MedicalAILayer()
    fps_buffer     = deque(maxlen=30)
    prev_time      = time.time()
    vitals_result  = VitalsResult()
    ai_result      = None
    baseline_dev   = {}
    show_history   = False
    ground_truth   = None
    gt_source      = "manual"
    paper_metrics  = {}
    bpm_series        = []
    last_valid_bpm    = 0.0
    last_valid_bpm_ts = 0.0
    display_bpm       = 0.0 
    BPM_HOLDOVER_SEC  = cfg.BPM_HOLDOVER_SEC
    gt_series      = []
    screenshot_n   = 0
    frame_count    = 0
    last_ai_check  = 0
    _cached_landmarks = None
    _mediapipe_every  = 2
    _last_mp_frame    = -1
    _motion_clear_frame = -1

    GRAPH_H  = 180
    SPEC_H   = 110
    VITALS_H = 95
    AI_H     = 100
    HIST_H   = 115
    HINT_H   = 24

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        vitals_engine.failure_log.tick()

        now       = time.time()
        dt        = now - prev_time
        prev_time = now
        fps_inst  = 1.0 / dt if dt > 1e-9 else 30.0
        fps_buffer.append(fps_inst)
        fps_avg   = float(np.mean(fps_buffer))

        vitals_engine.update_fps(fps_avg)
        frame = cv2.GaussianBlur(frame, (3, 3), 0)


        _mp_interval = 1 if frame_count < 90 else _mediapipe_every
        if frame_count % _mp_interval == 0:
            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = face_mesh.process(rgb)
            _cached_landmarks = (result.multi_face_landmarks[0]
                                 if result.multi_face_landmarks else None)
            _last_mp_frame = frame_count
        face_landmarks = _cached_landmarks
        face_detected  = face_landmarks is not None

        fr = fusion_engine.update(frame, face_landmarks, H, W)

        if fr.motion_rejected:
            vitals_engine.failure_log.log("motion_rejected", f"score={fr.motion_score:.2f}", fr.motion_score)
        if fr.exposure_drift > 35:
            vitals_engine.failure_log.log("exposure_drift", f"drift={fr.exposure_drift:.1f}", fr.exposure_drift)

        mean_brightness = fr.frame_brightness


        if fr.motion_rejected:
            _motion_clear_frame = -1
            holdover_age = now - last_valid_bpm_ts
            display_bpm = last_valid_bpm if (last_valid_bpm > 0 and holdover_age <= BPM_HOLDOVER_SEC) else display_bpm
        elif _motion_clear_frame < 0:

            _motion_clear_frame = frame_count
            display_bpm = last_valid_bpm if last_valid_bpm > 0 else display_bpm
        elif frame_count - _motion_clear_frame < _cfg.MOTION_GRACE_FRAMES:
            display_bpm = last_valid_bpm if last_valid_bpm > 0 else display_bpm
        elif fr.fused_bpm > 0:

            display_bpm       = fr.fused_bpm
            last_valid_bpm    = fr.fused_bpm
            last_valid_bpm_ts = now
        elif last_valid_bpm > 0 and (now - last_valid_bpm_ts) <= BPM_HOLDOVER_SEC:
            display_bpm = last_valid_bpm

        if not fr.motion_rejected:
            vitals_engine.ingest_signal(fr.chrom_signal, fr.fused_bpm, fr.fused_sqi)


        _vitals_ready = (
            frame_count % 15 == 0
            and len(fr.chrom_signal) >= vitals_engine.MIN_FRAMES
            and not fr.motion_rejected

        )
        if _vitals_ready:
            vitals_result = vitals_engine.compute(fr.chrom_signal, fr.fused_bpm, fr.fused_sqi)
            vitals_result.session_confidence = fr.session_confidence
            vitals_engine.update_baseline(vitals_result)
            baseline_dev = vitals_engine.baseline_deviation(vitals_result)

            session_hist.ingest(
                vitals_result.bpm, vitals_result.resp_rate,
                vitals_result.hrv_sdnn, vitals_result.stress_index,
                vitals_result.sqi, vitals_result.arrhythmia_flag
            )

            if display_bpm > 0 and fr.fused_sqi >= SQI_HARD_GATE:
                bpm_series.append(display_bpm)
                if ground_truth:
                    gt_series.append(ground_truth)

        if frame_count % 90 == 0 and display_bpm > 0 and fr.fused_sqi >= SQI_HARD_GATE and not fr.motion_rejected:
            if now - last_ai_check > 30:
                v_dict = {
                    "bpm": vitals_result.bpm, "resp_rate": vitals_result.resp_rate,
                    "hrv_sdnn": vitals_result.hrv_sdnn, "hrv_rmssd": vitals_result.hrv_rmssd,
                    "lf_hf_ratio": vitals_result.lf_hf_ratio,
                    "stress_index": vitals_result.stress_index,
                    "sqi": fr.fused_sqi,
                    "session_confidence": fr.session_confidence,
                    "arrhythmia_flag": vitals_result.arrhythmia_flag,
                    "arrhythmia_detail": vitals_result.arrhythmia_detail
                }
                trend    = session_hist.get_trend_summary()
                ai_result = ai_layer.interpret(v_dict, baseline_dev, trend)
                last_ai_check = now

        if frame_count % 90 == 0 and fr.fused_bpm > 0:
            bd = fr.sqi_breakdown
            print(f"\n[ROI DEBUG] frame={frame_count}  fused_BPM={fr.fused_bpm:.1f}"
                  f"fused_SQI={fr.fused_sqi:.1f}%  "
                  f"session_conf={fr.session_confidence:.1f}%  "
                  f"motion={'REJECT' if fr.motion_rejected else 'ok'}  "
                  f"exposure_drift={fr.exposure_drift:.1f}")
            for rname, roi in fr.roi_signals.items():
                if roi.valid:
                    skin_cov = getattr(roi, 'skin_coverage', 1.0)
                    agree_f  = getattr(roi, 'agreement_factor', 1.0)
                    print(f"  {rname:12s}  BPM={roi.bpm:6.1f}  SQI={roi.sqi:5.1f}%  "
                          f"dyn_w={roi.dynamic_weight:6.3f}  agree={agree_f:.2f}  "
                          f"bright={roi.roi_brightness:5.1f}  "
                          f"snr={roi.roi_snr:5.1f}  reg={roi.roi_regularity:5.1f}  "
                          f"skin_cov={skin_cov:.2f}")
            print(f"  SQI breakdown: SNR={bd.get('snr',0):.1f}%  "
                  f"reg={bd.get('regularity',0):.1f}%  var={bd.get('variance',0):.1f}%  "
                  f"stab={bd.get('temporal_stability',0):.1f}%  peak={bd.get('peak_consistency',0):.1f}%  "
                  f"motion_pen={bd.get('motion_penalty',0):.1f}%  "
                  f"light_pen={bd.get('lighting_penalty',0):.1f}%  "
                  f"fft_pen={bd.get('fft_penalty',0):.1f}%")

        tips = guidance_eng.check_webcam_quality(
            fps_avg, fr.motion_score, fr.fused_sqi, face_detected, mean_brightness
        )
        guidance_msg = guidance_eng.realtime_face_guidance(
            fr.motion_score, fr.fused_sqi, face_detected,
            mean_brightness, display_bpm
        )
        sqi_explanations = guidance_eng.sqi_explanation(
            fr.sqi_breakdown, fr.motion_score, mean_brightness
        )

        draw_roi_overlay(frame, fr.roi_signals, fr.motion_rejected)

        if fr.motion_rejected:
            put_text(frame, "MOTION — buffering paused (hysteresis active)",
                     W // 2 - 160, 75, 0.52, PALETTE["orange"], 2)

        draw_arrhythmia_warning(frame, vitals_result, H, W)
        draw_guidance(frame, tips, guidance_msg, sqi_explanations, H, W,
                      exposure_drift=fr.exposure_drift)

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (W, 72), PALETTE["panel"], -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

        if display_bpm > 0:

            sqi = fr.fused_sqi
            

            if fr.fused_bpm == 0 and fr.n_valid_rois > 0:
                put_text(frame, "BPM: Analyzing...", 20, 44, 1.2, PALETTE["yellow"], 2)
                put_text(frame, f"Signal low quality ({sqi:.0f}%) — stabilize...", 20, 62, 0.45, PALETTE["orange"])
            else:
                if sqi >= 55:
                    bpm_c = PALETTE["green"]
                elif sqi >= 35:
                    bpm_c = PALETTE["yellow"]
                elif sqi >= 18:
                    bpm_c = PALETTE["orange"]
                else:
                    bpm_c = PALETTE["red_bright"]
                lock_icon = " 🔒" if fr.bpm_locked else ""
                put_text(frame, f"BPM: {display_bpm:.0f}{lock_icon}", 20, 44, 1.2, bpm_c, 2)
                cl   = vitals_result.confidence_label
                cl_c = PALETTE["green"] if cl == "GOOD" else PALETTE["yellow"] if cl == "FAIR" else PALETTE["red"]
                put_text(frame, f"conf: {fr.fused_sqi:.0f}%  [{cl}]  Session:{fr.session_confidence:.0f}%",
                         20, 62, 0.45, cl_c)

            bd   = fr.sqi_breakdown
            if bd:
                why_parts = []
                if bd.get("motion_penalty", 0) > 15:
                    why_parts.append(f"motion-{bd['motion_penalty']:.0f}%")
                if bd.get("lighting_penalty", 0) > 15:
                    why_parts.append(f"light-{bd['lighting_penalty']:.0f}%")
                if bd.get("fft_penalty", 0) > 5:
                    why_parts.append(f"fft-{bd['fft_penalty']:.0f}%")
                if bd.get("snr", 100) < 35:
                    why_parts.append(f"SNR-low")
                if bd.get("regularity", 100) < 30:
                    why_parts.append(f"irreg")
                if why_parts:
                    put_text(frame, "WHY: " + " | ".join(why_parts),
                             20, 72, 0.38, PALETTE["gray"])

            bdev_label = baseline_dev.get("bpm_label", "")
            if bdev_label and bdev_label != "NORMAL":
                bdev_c = PALETTE["orange"] if "ELEVATED" in bdev_label else PALETTE["cyan"]
                put_text(frame, f"vs baseline: {bdev_label}  z={baseline_dev.get('bpm_z',0):+.1f}",
                         W // 2 - 130, 22, 0.45, bdev_c)

        elif fr.fused_bpm > 0 and fr.fused_sqi < SQI_HARD_GATE:


            _show_bpm   = fr.fused_bpm
            display_bpm = fr.fused_bpm
            last_valid_bpm    = fr.fused_bpm
            last_valid_bpm_ts = now
            put_text(frame, f"BPM: {_show_bpm:.0f}", 20, 44, 1.2,
                     PALETTE["orange"], 2)
            put_text(frame, f"low signal  SQI:{fr.fused_sqi:.0f}%",
                     20, 58, 0.38, PALETTE["orange"])
            if sqi_explanations:
                put_text(frame, "WHY: " + " | ".join(sqi_explanations[:2]),
                         20, 68, 0.38, PALETTE["gray"])
        else:
            n_bufs = max(len(v.buf_g) for v in fusion_engine.rois.values())
            pct    = min(int(n_bufs / MIN_FRAMES * 100), 100)
            put_text(frame, f"Collecting... {min(n_bufs, MIN_FRAMES)}/{MIN_FRAMES}  ({pct}%)", 20, 42, 0.75, PALETTE["gray"])

        skin = fusion_engine.skin_calibrator.skin_type_label()
        put_text(frame, f"Skin: {skin}", W - 230, 22, 0.4, PALETTE["gray"])
        put_text(frame, f"FPS: {fps_avg:.1f}", W - 100, 22, 0.45, PALETTE["gray"])

        yaw, pitch, pose_ok = fusion_engine._last_pose
        if not pose_ok:
            put_text(frame, f"HEAD POSE: yaw={yaw:.0f}° pitch={pitch:.0f}° — straighten up",
                     W // 2 - 200, 82, 0.45, PALETTE["orange"])
        elif abs(yaw) > 15 or abs(pitch) > 15:
            put_text(frame, f"Pose: yaw={yaw:.0f}° pitch={pitch:.0f}°",
                     W - 270, 38, 0.38, PALETTE["gray"])

        jaw = fusion_engine._last_jaw_result
        if jaw.get("talking"):
            put_text(frame, "TALKING — cheeks suppressed", W // 2 - 140, 62, 0.45, PALETTE["yellow"])
        elif jaw.get("blinking"):
            put_text(frame, "BLINK", W - 80, 38, 0.38, PALETTE["gray"])

        if fr.brightness_normalized:
            put_text(frame, f"EXP-DRIFT:{fr.exposure_drift:.0f}", W - 100, 38, 0.38, PALETTE["teal"])

        trend_data = fusion_engine.conf_trend.get_trend()
        if trend_data.get("trend") == "DEGRADING":
            put_text(frame, f"SQI TREND: DEGRADING ({trend_data.get('slope_per_sec',0):+.1f}/s)",
                     20, 82, 0.38, PALETTE["orange"])
        elif trend_data.get("trend") == "IMPROVING":
            put_text(frame, f"SQI TREND: IMPROVING",
                     20, 82, 0.38, PALETTE["green_dim"])

        status_c = PALETTE["green"] if face_detected else PALETTE["red"]
        cv2.circle(frame, (W - 18, 18), 7, status_c, -1)

        graph_canvas  = np.zeros((GRAPH_H, W, 3), dtype=np.uint8)
        spec_canvas   = np.zeros((SPEC_H,  W, 3), dtype=np.uint8)
        vitals_canvas = np.zeros((VITALS_H, W, 3), dtype=np.uint8)
        ai_canvas     = np.zeros((AI_H,    W, 3), dtype=np.uint8)
        hist_canvas   = np.zeros((HIST_H,  W, 3), dtype=np.uint8)
        hint_canvas   = np.full((HINT_H,   W, 3), PALETTE["panel"], dtype=np.uint8)

        method_stats = fusion_engine.arbitrator.get_win_stats()
        winner_label = max(method_stats, key=method_stats.get).upper()


        _t3 = fr.top3_peaks if fr.top3_peaks else []
        _t3_str = "  |  Peaks: " + "  ".join(
            f"#{i+1} {bpm:.0f}BPM(pw={pw:.3f})" for i, (bpm, pw) in enumerate(_t3[:3])
        ) if _t3 else ""
        _subharm_str = "  ⚠ SUBHARM→corrected" if fr.subharm_corrected else ""

        draw_signal_graph(
            graph_canvas, fr.chrom_signal, GRAPH_H, W,
            color=PALETTE["green"],
            label=f"Method:{winner_label} (CHROM:{method_stats['chrom']:.0%} POS:{method_stats['pos']:.0%} G:{method_stats['green']:.0%})"
                  f"  |  Motion:{fr.motion_score:.1f}  |  SQI-trend:{trend_data.get('trend','?')}"
                  f"  |  State:{getattr(fr,'physio_state','?')} RR:{getattr(fr,'resp_rate_bpm',0):.0f}br/min CF:{getattr(fr,'cardiac_fraction',1.0):.2f}"
                  f"{_t3_str}{_subharm_str}",
            bpm=fr.fused_bpm, sqi=fr.fused_sqi, sqi_breakdown=fr.sqi_breakdown,
            session_confidence=fr.session_confidence
        )

        draw_spectrum_graph(spec_canvas, fr.freqs, fr.power, fr.fused_bpm, SPEC_H, W)
        draw_vitals_panel(vitals_canvas, vitals_result, baseline_dev, VITALS_H, W)
        draw_ai_panel(ai_canvas, ai_result, AI_H, W)

        if show_history:
            history_rows = session_hist.get_history(limit=5)
            trend        = session_hist.get_trend_summary()
            draw_history_panel(hist_canvas, history_rows, trend, HIST_H, W)
        else:
            draw_panel(hist_canvas, 0, 0, W, HIST_H)
            if paper_metrics:
                ba   = paper_metrics.get("bland_altman", {})
                pe   = paper_metrics.get("pearson_gt", {})
                basic = f"MAE:{paper_metrics.get('MAE','?')}  RMSE:{paper_metrics.get('RMSE','?')}"
                ba_str = f"  BA_bias:{ba.get('mean_diff_bpm','?')}  LOA:[{ba.get('loa_lower','?')},{ba.get('loa_upper','?')}]" if ba else ""
                pe_str = f"  r={pe.get('pearson_r','?')}" if pe else ""
                put_text(hist_canvas, f"METRICS: {basic}{ba_str}{pe_str}",
                         10, HIST_H // 2 + 6, 0.38, PALETTE["cyan"])
            else:
                put_text(hist_canvas,
                         "[H] history  [M] paper metrics  [A] AI  [G] ground truth  [P] pulse ox  [F] failure log",
                         10, HIST_H // 2 + 6, 0.38, PALETTE["gray"])

        hint_text = "[Q]quit [R]reset [S]save [A]AI [H]hist [M]metrics [G]GT [P]pulseox [D]diag [L]log [Z]screenshot"
        put_text(hint_canvas, hint_text, 8, 17, 0.36, PALETTE["gray_dim"])

        combined = np.vstack([
            frame,
            graph_canvas,
            spec_canvas,
            vitals_canvas,
            ai_canvas,
            hist_canvas,
            hint_canvas,
        ])

        cv2.imshow("rPPG Research System v5.0", combined)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == 27:
            print("\n[INFO] Saving session before exit...")
            session_hist.save_session(notes="auto-saved on exit")
            break
        elif key == ord('r'):
            fusion_engine.reset()
            vitals_result = VitalsResult()
            bpm_series.clear()
            last_valid_bpm    = 0.0
            last_valid_bpm_ts = 0.0
            display_bpm       = 0.0
            gt_series.clear()
            print("[INFO] Buffers reset.")
        elif key == ord('s'):
            sid = session_hist.save_session()
            print(f"[INFO] Session saved. ID: {sid}")
        elif key == ord('a'):
            if fr.fused_bpm > 0:
                v_dict = {
                    "bpm": vitals_result.bpm, "resp_rate": vitals_result.resp_rate,
                    "hrv_sdnn": vitals_result.hrv_sdnn, "hrv_rmssd": vitals_result.hrv_rmssd,
                    "lf_hf_ratio": vitals_result.lf_hf_ratio,
                    "stress_index": vitals_result.stress_index,
                    "sqi": fr.fused_sqi,
                    "session_confidence": fr.session_confidence,
                    "arrhythmia_flag": vitals_result.arrhythmia_flag,
                    "arrhythmia_detail": vitals_result.arrhythmia_detail
                }
                ai_result = ai_layer.interpret(v_dict, baseline_dev, session_hist.get_trend_summary())
                last_ai_check = time.time()
                print("[INFO] AI analysis triggered.")
        elif key == ord('h'):
            show_history = not show_history
        elif key == ord('m'):
            paper_metrics = vitals_engine.get_paper_metrics(
                ground_truth_bpm=ground_truth,
                estimated_bpm_series=bpm_series if len(bpm_series) > 5 else None,
                ground_truth_series=gt_series  if len(gt_series)  > 5 else None
            )
            print(f"[METRICS] {paper_metrics}")
        elif key == ord('g'):
            try:
                gt_input = input("Enter BPM (manual count): ").strip()
                ground_truth = float(gt_input)
                vitals_engine.ground_truth.set_ground_truth(ground_truth, source="manual")
                print(f"[GT] Manual BPM set: {ground_truth}")
            except Exception:
                print("[INFO] Invalid input.")
        elif key == ord('p'):
            try:
                gt_input = input("Enter pulse oximeter BPM reading: ").strip()
                ground_truth = float(gt_input)
                gt_source = "pulse_oximeter"
                vitals_engine.ground_truth.set_ground_truth(ground_truth, source="pulse_oximeter")
                print(f"[GT] Pulse oximeter BPM: {ground_truth}")
            except Exception:
                print("[INFO] Invalid input.")
        elif key == ord('t'):
            tests = ["skin_tone", "distance", "lighting", "posture", "breathing", "stress"]
            print(f"Robustness tests: {tests}")
            try:
                t_name = input("Enter test name: ").strip()
                vitals_engine.start_robustness_test(t_name)
            except Exception:
                pass
        elif key == ord('f'):
            summary = vitals_engine.failure_log.get_summary()
            print(f"\n[FAILURE MODES] {summary}")
            recent = vitals_engine.failure_log.recent_events(10)
            for ev in recent:
                ts_str = time.strftime("%H:%M:%S", time.localtime(ev["ts"]))
                print(f"  {ts_str}  {ev['type']:25s}  {ev['detail']}")
        elif key == ord('d'):
            diag = fusion_engine.get_diagnostics()
            print("\n[DIAGNOSTICS]")
            for k, v in diag.items():
                print(f"  {k}: {v}")
        elif key == ord('l'):
            if fusion_engine.repro_logger:
                out = fusion_engine.repro_logger.save()
                print(f"[REPRO] Saved to: {out}")
            else:
                print("[REPRO] Logging not enabled. Run with: python rppg_main.py --log")
        elif key == ord('z'):
            screenshot_n += 1
            fname = f"rppg_research_{screenshot_n:03d}.png"
            cv2.imwrite(fname, combined)
            print(f"[INFO] Screenshot: {fname}")

    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()
    print("[INFO] Done.\n")

if __name__ == "__main__":
    main()
