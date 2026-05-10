"""
rppg_config.py  –  Central configuration system
All constants live here. Import: from rppg_config import cfg
"""

from dataclasses import dataclass


@dataclass
class RPPGConfig:
    # ── Buffer / timing ──────────────────────────────────────────────────────
    BUFFER_SIZE: int   = 150   # 5s at 30fps – faster pipeline response
    FPS_TARGET:  float = 30.0
    MIN_FRAMES:  int   = 60   # 2s – faster first estimate
    CALIBRATION_FRAMES: int = 150
    RPPG_ALGO: str = "POS"  # Default algorithm: POS, CHROM, GREEN, ICA

    # ── BPM physiological bounds ──────────────────────────────────────────────
    BPM_LOW:  float = 42.0
    BPM_HIGH: float = 200.0

    # ── Respiration ──────────────────────────────────────────────────────────
    RESP_LOW:  float = 0.10
    RESP_HIGH: float = 0.50

    # ── SQI gating ───────────────────────────────────────────────────────────
    SQI_HARD_GATE:          float = 20.0   # v12: lower = more output, color = confidence
    SQI_DISPLAY_GATE:       float = 10.0   # v12: show if any signal exists
    BASELINE_SQI_THRESHOLD: float = 45.0
    ARRHYTHMIA_SQI_GATE:    float = 65.0   # Fix #8

    # ── SQI component weights (Rebalanced per "Real Talk" feedback) ─────────
    # v12 color bands: GREEN>=55, YELLOW>=35, ORANGE>=18, RED>0
    SQI_WEIGHT_SNR:          float = 0.35
    SQI_WEIGHT_REGULARITY:   float = 0.25
    SQI_WEIGHT_TEMPORAL:     float = 0.15
    SQI_WEIGHT_PEAK_CONSIST: float = 0.15
    SQI_WEIGHT_VARIANCE:     float = 0.10
    SQI_PENALTY_WEIGHT:      float = 0.10   # Penalty is now a component, not a dictator
    SQI_MAX_MOTION_PENALTY:  float = 0.45
    SQI_MAX_LIGHTING_PENALTY: float = 0.40
    SQI_MAX_FFT_PENALTY:     float = 0.45   # FFT sharpness is the main quality gate

    # ── Fix A: Agreement → hard SQI penalty ──────────────────────────────────
    AGREEMENT_SQI_PENALTY_BELOW: float = 0.50  # align with ROI_MIN_AGREEMENT_FACTOR
    AGREEMENT_SQI_PENALTY_COEFF: float = 0.30  # v12: soft penalty, not blocking

    # ── Fix B: Dynamic weight clamping ────────────────────────────────────────
    DYN_WEIGHT_MIN: float = 0.3
    DYN_WEIGHT_MAX: float = 3.0
    # After normalizing, each ROI weight is bounded to avoid dominance
    DYN_WEIGHT_NORM_MAX: float = 0.75   # max fraction any single ROI can hold

    # ── Fix C: BPM outlier threshold (tighter than before) ───────────────────
    FUSION_OUTLIER_THRESHOLD: float = 10.0   # base: +10.0 added = 20 BPM from ref-BPM

    # ── Fix D: Regularity hard gate ───────────────────────────────────────────
    REGULARITY_HARD_GATE: float = 0.0    # v12: disabled – reg is a weight, not gate

    # ── Fix E: FFT adaptive parameters ───────────────────────────────────────
    FFT_DOMINANCE_STRONG:  float = 0.30  # above → no penalty
    FFT_DOMINANCE_MEDIUM:  float = 0.15  # above → small penalty
    FFT_DOMINANCE_WEAK:    float = 0.08  # above → medium penalty
                                          # below → heavy penalty

    # ── Fix F: BPM velocity constraint ───────────────────────────────────────
    BPM_VELOCITY_SOFT_LIMIT: float = 5.0    # tighter velocity gate
    BPM_VELOCITY_DAMP_ALPHA:  float = 0.20  # stronger pull toward prev (was 0.35)

    # ── Fix G: Session confidence ─────────────────────────────────────────────
    SESSION_CONF_WEIGHT_SQI:         float = 0.40  # SQI is most important
    SESSION_CONF_WEIGHT_AGREEMENT:   float = 0.30  # agreement mandatory
    SESSION_CONF_WEIGHT_TEMPORAL:    float = 0.20
    SESSION_CONF_WEIGHT_MOTION:      float = 0.10
    SESSION_CONF_CONSISTENCY_WINDOW: int   = 60
    # Hard multiplier: if agreement=0 → conf halved minimum
    SESSION_CONF_AGREEMENT_FLOOR:    float = 0.4

    # ── Fix H: Exposure drift SQI penalty ────────────────────────────────────
    EXPOSURE_DRIFT_WARN:        float = 8.0
    EXPOSURE_DRIFT_FREEZE:      float = 20.0
    EXPOSURE_DRIFT_SQI_PENALTY: float = 0.35  # max SQI reduction from drift
    EXPOSURE_DRIFT_WEIGHT_MULT: float = 0.45

    # ── Dynamic weight kill switches ─────────────────────────────────────────
    SNR_KILL_THRESHOLD: float = 20.0
    SNR_KILL_WEIGHT:    float = 0.08
    AGREEMENT_KILL_BELOW: float = 0.55   # earlier kill (was 0.50)
    AGREEMENT_KILL_MULT:  float = 0.10   # harder crush (was 0.15)

    # ── Fix 3: Real agreement computation (exp-based) ─────────────────────────
    AGREEMENT_STD_K: float = 12.0   # Bug2 fix: std=12→0.37, more realistic
    AGREEMENT_DEV_K: float = 9.0    # Bug2 fix: ~9 BPM tolerance (exp(-8/9)=0.41)

    # ── Fix 2: Physiological BPM plausibility gate ────────────────────────────
    BPM_PLAUSIBLE_LOW:  float = 42.0   # below this in non-athlete → reject ROI
    BPM_PLAUSIBLE_HIGH: float = 140.0  # lowered: 140 catches most resting+light activity

    # ── Fix 4: Single ROI confidence downgrade ────────────────────────────────
    SINGLE_ROI_SQI_MULT: float = 1.00  # v12: no penalty – single ROI is valid

    # ── Fix H: Harmonic suspect weight penalty ────────────────────────────────
    HARMONIC_SUSPECT_MULT: float = 0.05  # near-zero discard for harmonic ROI

    # ── FIX A: Hard agreement gate ───────────────────────────────────────────
    ROI_MIN_AGREEMENT_FACTOR: float = 0.20  # v12: allow wide deviation, weight handles it
    # At AGREEMENT_DEV_K=5: dev≥4 BPM → factor=0.45 → hard rejected
    # This means ROI must be within ~4 BPM of reference to survive

    # ── Temporal Trust Lock (Priority 2) ─────────────────────────────────────
    STABLE_LOCK_SECONDS:   float = 5.0   # FIX4: 8s was unrealistic; 5s is sufficient for rPPG convergence
    STABLE_LOCK_BPM_STD:   float = 5.0   # Bug6 fix: slightly more tolerant
    STABLE_HOLD_ALPHA:     float = 0.05  # when locked: very slow EMA (almost frozen)

    # ── ROI Auto-Exclusion (Priority 2) ──────────────────────────────────────
    ROI_BAD_STREAK_LIMIT:  int   = 120   # v12: 4s before exclusion (was 1s)
    ROI_BAD_SQI_THRESH:    float = 15.0  # v12: only exclude truly dead ROI
    ROI_REHAB_FRAMES:      int   = 30    # v12: quick rehab – try again fast

    # ── Fix C: Multi-window FFT voting ──────────────────────────────────────
    FFT_WINDOW_SEC:        float = 4.0   # FIX5: 2s → 4s; resolves 40-50 BPM ambiguity (Δf=0.25Hz@4s vs 0.5Hz@2s)
    FFT_WINDOW_VOTE_TOL:   float = 0.10  # Hz tolerance for "same peak" across windows
    FFT_HISTORY_OVERRIDE:  float = 0.18  # Hz threshold to override new peak with history

    # ── Peak Temporal Bonus (Priority 2) ─────────────────────────────────────
    PEAK_TEMPORAL_BONUS:   float = 0.25  # bonus weight for peak near prev frame peak
    PEAK_TEMPORAL_TOL_HZ:  float = 0.12  # tolerance for "near previous peak" in Hz
    TEMPORAL_STABILITY_WINDOW: int = 10  # last 10 BPMs for stability check

    # ── Fix T: Per-ROI temporal spike penalty ─────────────────────────────────
    ROI_TEMPORAL_SPIKE_BPM:  float = 15.0   # jump > this → penalize (was 20)
    ROI_TEMPORAL_SPIKE_MULT: float = 2.0    # penalty scale factor (was 1.5)

    # ── Fix BP: Tight bandpass (already in core as FREQ_LOW_TIGHT/HIGH_TIGHT) ─
    # Documented here for reference: 0.8–2.5 Hz = 48–150 BPM

    # ── Motion ────────────────────────────────────────────────────────────────
    MOTION_ENTER_THRESHOLD:  float = 2.8   # FIX1: raised – micro-movement (breathing/tremor ~0.5-1.5px) no longer rejects
    MOTION_EXIT_THRESHOLD:   float = 1.8   # FIX1: raised to match new enter threshold (hysteresis gap = 1.0)
    BPM_HOLDOVER_SEC:        float = 5.0   # how long to show last good BPM after motion
    MOTION_GRACE_FRAMES:     int   = 12    # longer grace: signal needs time to recover

    # ── BPM smoother ─────────────────────────────────────────────────────────
    BPM_EMA_ALPHA:     float = 0.20   # Bug8 fix: faster convergence (was 0.12)
    BPM_MEDIAN_WINDOW: int   = 11     # Bug8 fix: faster window (was 17)
    BPM_MAX_JUMP:      float = 25.0   # Bug3 fix: smoother handles fine damping

    # ── Physiological plausibility ────────────────────────────────────────────
    TEMPORAL_BPM_MAX_JUMP_3FRAMES: float = 15.0  # physiological max change in 3 frames
    TEMPORAL_HISTORY_LEN: int = 7    # longer history (was 5)

    # ── Hierarchical fusion ───────────────────────────────────────────────────
    CLUSTER_DISTANCE_BPM: float = 14.0  # Bug5 fix: forehead+cheek 63/72 → same cluster
    MIN_CLUSTER_WEIGHT:   float = 0.20

    # ── Calibration ───────────────────────────────────────────────────────────
    CALIBRATION_SQI_FLOOR: float = 40.0

    # ── ROI brightness gating ────────────────────────────────────────────────
    ROI_BRIGHTNESS_MIN: float = 45.0
    ROI_BRIGHTNESS_MAX: float = 205.0


cfg = RPPGConfig()
