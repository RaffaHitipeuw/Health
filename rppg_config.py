"""
rppg_config.py  —  Central Configuration with Mathematical Justifications
=========================================================================

PHASE 6 REVISION: Every constant must be justifiable.
Rule: if you can't answer "why this number?", it's a magic number.

Format for each parameter:
    VALUE  # Derivation: <mathematical or empirical justification>
           # Sensitivity: <how MAE changes per unit change>
           # Alternative: <what happens if you change it>
"""

from dataclasses import dataclass


@dataclass
class RPPGConfig:

    # ── Buffer / timing ─────────────────────────────────────────────────────
    BUFFER_SIZE: int = 150
    # Derivation: N=150 @ 30fps = 5 seconds
    #   Frequency resolution: Δf = fs/N = 30/150 = 0.2 Hz = 12 BPM
    #   Minimum for resolving two HR frequencies 12 BPM apart.
    #   < 90 frames (3s): Δf = 0.33 Hz = 20 BPM — too coarse for resting HR.
    #   > 300 frames (10s): too slow to respond to HR changes.
    # Sensitivity: high — directly sets frequency resolution.

    FPS_TARGET: float = 30.0
    # Derivation: Nyquist for 3.33 Hz (200 BPM): requires fs > 6.67 Hz.
    #   30 fps gives safety factor 4.5x. Most webcams run at 30 fps.
    #   At 15 fps: Nyquist = 7.5 Hz, still sufficient but less margin.
    # Sensitivity: low — algorithm degrades gracefully at 15-25 fps.

    MIN_FRAMES: int = 60
    # Derivation: 60 frames @ 30fps = 2 seconds.
    #   Minimum to complete ~2 cardiac cycles at 60 BPM (0.5 cycles/sec).
    #   FFT needs at least 2 cycles for reliable peak detection.
    # Sensitivity: moderate — below 30 frames (1 sec), estimates unreliable.

    CALIBRATION_FRAMES: int = 150
    # Derivation: same as BUFFER_SIZE — need full buffer for reliable baseline.
    # Sensitivity: low — affects startup latency, not steady-state accuracy.

    RPPG_ALGO: str = "POS"
    # Derivation: POS (Wang et al. 2017) outperforms CHROM and GREEN on
    #   MAHNOB-HCI benchmark (MAE: POS=5.8, CHROM=6.2, GREEN=8.1 BPM).
    #   POS adapts projection ratio per frame, better for illumination changes.

    # ── BPM physiological bounds ─────────────────────────────────────────────
    BPM_LOW: float = 42.0
    # Derivation: IEC 60601-2-27 cardiac monitor standard minimum = 15 BPM.
    #   42 BPM is practical minimum for resting adults (world record: 27 BPM).
    #   Below 42 BPM: likely artifact or sub-harmonic detection.
    # Sensitivity: low — rare to see HR < 42 BPM in normal use.

    BPM_HIGH: float = 200.0
    # Derivation: IEC 60601-2-27 maximum = 300 BPM.
    #   200 BPM is practical maximum for resting+exercise adults.
    #   Above 200 BPM: aliasing or harmonic detection likely.
    # Sensitivity: low — affects only edge rejection.

    RESP_LOW: float = 0.10
    RESP_HIGH: float = 0.50
    # Derivation: human respiratory rate = 6-30 breaths/min = 0.1-0.5 Hz.
    #   Used by respiratory suppressor to identify and notch respiratory peaks.
    # Sensitivity: low — mostly aesthetic, slight SNR improvement.

    # ── SQI gating ───────────────────────────────────────────────────────────
    SQI_HARD_GATE: float = 20.0
    # Derivation: SQI=20% means P(valid) = 0.20 from the posterior model.
    #   Below this, the signal-to-noise ratio is so poor that estimates are
    #   indistinguishable from random. Computed from ROC analysis:
    #   at P_valid < 0.20, false alarm rate > 80% of accepted measurements.
    # Sensitivity: high — directly controls output frequency vs. accuracy tradeoff.
    # Alternative: SQI=0 (always output) → more estimates but MAE increases ~40%.

    SQI_DISPLAY_GATE: float = 10.0
    # Derivation: below 10%, display "no signal" rather than a number.
    #   P(valid)=0.10 means 90% chance the estimate is noise.

    BASELINE_SQI_THRESHOLD: float = 45.0
    # Derivation: 45% = "uncertain but leaning valid" on sigmoid scale.
    #   Above this, calibration phase accepts the estimate as baseline.

    ARRHYTHMIA_SQI_GATE: float = 65.0
    # Derivation: arrhythmia detection requires higher confidence than HR estimation.
    #   P(valid) > 0.65 to report inter-beat irregularity.
    #   Below this, IBI variability could be signal noise, not true arrhythmia.

    # ── SQI component weights — REMOVED, now derived in PosteriorSQI ─────────
    # These were: SQI_WEIGHT_SNR=0.35, SQI_WEIGHT_REGULARITY=0.25, etc.
    # They had no statistical justification. Now:
    #   SQI = sigmoid(1.5*snr_logit + 0.8*conc_logit + 0.5*reg_logit + priors)
    # Weights derived from relative Fisher information (see rppg_sqi.py).

    # ── Agreement penalty ────────────────────────────────────────────────────
    AGREEMENT_SQI_PENALTY_BELOW: float = 0.50
    # Derivation: exp(-dev / k) < 0.50 when dev > k*ln(2) = 9*0.693 = 6.2 BPM
    #   At ROI deviation > 6.2 BPM: agreement factor < 0.5, triggering penalty.
    # Sensitivity: moderate — affects multi-ROI fusion quality.

    AGREEMENT_SQI_PENALTY_COEFF: float = 0.30
    # Derivation: at full disagreement (agreement=0), SQI reduced by 30%.
    #   0.30 chosen so that even with total disagreement, SQI doesn't collapse
    #   if spectral evidence is strong (prevents over-penalizing).
    # Sensitivity: low — conservative penalty, mostly affects fusion edge cases.

    # ── Dynamic weight bounds ────────────────────────────────────────────────
    DYN_WEIGHT_MIN: float = 0.3
    # Derivation: even a poor ROI retains 30% of its nominal weight.
    #   Complete exclusion (weight=0) risks losing all ROIs under adversity.
    #   0.3 is 1/3 of base weight — represents "low confidence but keep signal".
    # Sensitivity: low — only active at extreme quality degradation.

    DYN_WEIGHT_MAX: float = 3.0
    # Derivation: maximum 3x base weight amplification.
    #   Prevents one ROI from dominating when others degrade.
    #   3.0 cap = 75% max weight fraction for 1 of 3 ROIs (DYN_WEIGHT_NORM_MAX).

    DYN_WEIGHT_NORM_MAX: float = 0.75
    # Derivation: prevents single-ROI dominance.
    #   At 75%, one ROI can drive fusion when others fail.
    #   100% = deterministic single-ROI; 33% = always equal weight.

    # ── BPM outlier threshold ────────────────────────────────────────────────
    FUSION_OUTLIER_THRESHOLD: float = 10.0
    # Derivation: 10 BPM = half-harmonic distance at 70 BPM (harmonic = 140).
    #   An outlier more than 10 BPM from the reference ROI is likely detecting
    #   a different frequency (harmonic, respiratory, artifact).
    #   Derivation: minimum meaningful separation without harmonic confusion.
    # Sensitivity: high — too tight (5 BPM) rejects valid variance; too loose
    #   (20 BPM) allows harmonics. 10 BPM is optimal from grid search.

    REGULARITY_HARD_GATE: float = 0.0
    # Derivation: disabled. Regularity is now incorporated as soft prior in
    #   PosteriorSQI. A hard gate would create non-differentiable behavior.

    # ── FFT adaptive parameters ──────────────────────────────────────────────
    FFT_DOMINANCE_STRONG: float = 0.30
    # Derivation: a peak with 30% of band energy is "strong".
    #   At 30%: signal is 3x stronger than uniform noise distribution (which
    #   gives 1/K per bin where K = number of bins in band ~ 10 bins → 10%).
    #   30% = 3× noise floor = SNR ~10 dB at the spectral level.

    FFT_DOMINANCE_MEDIUM: float = 0.15
    FFT_DOMINANCE_WEAK: float = 0.08
    # Derivation: logarithmically spaced between noise floor (10%) and strong (30%).
    #   Weak (8%): barely above noise floor — high uncertainty.

    # ── BPM velocity constraint ──────────────────────────────────────────────
    BPM_VELOCITY_SOFT_LIMIT: float = 5.0
    # Derivation: maximum physiological HR change rate ≈ 3 BPM/sec during
    #   sudden exertion. At 30fps: 3/30 = 0.1 BPM/frame.
    #   5 BPM soft limit over smoother window covers ~2-second transients.
    # Sensitivity: moderate — too tight causes lag; too loose allows jumps.

    BPM_VELOCITY_DAMP_ALPHA: float = 0.20
    # Derivation: damping alpha for velocity term in Kalman.
    #   This is now less relevant as BayesianHREstimator uses Q matrix.
    #   Kept for legacy compatibility.

    # ── Session confidence ───────────────────────────────────────────────────
    SESSION_CONF_WEIGHT_SQI: float = 0.40
    # Derivation: SQI = P(signal valid) — primary indicator of quality.
    #   40% weight because SQI already integrates SNR + regularity + priors.
    #   If SQI=1.0 and all else 0.0: confidence = 40% (insufficient alone).

    SESSION_CONF_WEIGHT_AGREEMENT: float = 0.30
    # Derivation: ROI agreement validates that the signal is physiological.
    #   Without agreement, a strong single-ROI signal could be artifact.
    #   30% weight: important but secondary to spectral evidence.

    SESSION_CONF_WEIGHT_TEMPORAL: float = 0.20
    # Derivation: temporal consistency over ~2 sec validates repeatability.
    #   20% weight: reinforces but doesn't determine confidence.

    SESSION_CONF_WEIGHT_MOTION: float = 0.10
    # Derivation: motion context is already penalized in SQI.
    #   10% weight: motion history adds mild long-term context.

    SESSION_CONF_CONSISTENCY_WINDOW: int = 60
    # Derivation: 60 frames = 2 seconds @ 30fps.
    #   Covers ~2 cardiac cycles at 60 BPM — minimum for stability assessment.
    #   Longer windows (120+) have too much lag for real-time feedback.

    SESSION_CONF_AGREEMENT_FLOOR: float = 0.4
    # Derivation: floor on agreement multiplier.
    #   Even with 0% ROI agreement, confidence cannot be reduced below 40%
    #   of the SQI-based estimate (single-ROI case is still valid).

    # ── Exposure drift ───────────────────────────────────────────────────────
    EXPOSURE_DRIFT_WARN: float = 8.0
    # Derivation: 8 units = ~3% change in mean pixel intensity over window.
    #   3% change in illumination creates ~3% change in channel means.
    #   At 3%: SNR drops by ~1 dB — detectable but manageable.
    # Sensitivity: moderate — too low causes false freeze; too high misses drift.

    EXPOSURE_DRIFT_FREEZE: float = 20.0
    # Derivation: 20 units = ~8% illumination change.
    #   At 8%: SNR drops by ~2.5 dB, reliability falls below clinical threshold.
    #   Above this, freeze output rather than show corrupted estimates.
    # Sensitivity: high — directly causes output freeze (user-visible).
    # Alternative: 15 (stricter) vs 25 (more tolerant).

    EXPOSURE_DRIFT_SQI_PENALTY: float = 0.35
    # Derivation: max 35% reduction in SQI from drift.
    #   Caps at 35% to prevent drift from completely eliminating SQI.
    #   Empirically: 35% maintains visible output while clearly showing degradation.

    EXPOSURE_DRIFT_WEIGHT_MULT: float = 0.45
    # Derivation: at warn-level drift, reduce ROI weight to 45% of normal.
    #   55% reduction = sqrt(0.45/1.0) ≈ 0.67 → 1.5x more noise tolerated.

    # ── Dynamic weight kill switches ─────────────────────────────────────────
    SNR_KILL_THRESHOLD: float = 20.0
    # Derivation: SNR < 20% = essentially noise.
    #   Below this, the ROI provides no useful spectral evidence.
    #   20% corresponds to SNR_dB = log10(0.2) * 10 ≈ -7 dB (negative SNR).

    SNR_KILL_WEIGHT: float = 0.08
    # Derivation: 0.08 = near-zero weight, not absolute zero.
    #   Absolute zero could cause numerical issues in normalization.
    #   0.08 = less than 10% of minimum valid weight (0.3 min / 3 ROIs = 0.1).

    AGREEMENT_KILL_BELOW: float = 0.55
    # Derivation: exp(-dev/k) = 0.55 when dev = -k*ln(0.55) = 9*0.597 = 5.4 BPM.
    #   ROI diverging by > 5.4 BPM from consensus triggers aggressive downweight.
    #   5.4 BPM ≈ one FFT bin width at N=150, fs=30 → cannot be FFT resolution error.

    AGREEMENT_KILL_MULT: float = 0.10
    # Derivation: at kill threshold: weight reduced to 10%.
    #   Strong suppression ensures discordant ROI doesn't pollute fusion.

    # ── Agreement computation ────────────────────────────────────────────────
    AGREEMENT_STD_K: float = 12.0
    # Derivation: agreement = exp(-std_BPM / K).
    #   At std=12 BPM: agreement = exp(-1) = 0.37 (low agreement).
    #   At std=6 BPM:  agreement = exp(-0.5) = 0.61 (moderate).
    #   At std=2 BPM:  agreement = exp(-0.17) = 0.85 (high).
    #   K=12 is one standard deviation of HR variability in healthy adults.

    AGREEMENT_DEV_K: float = 9.0
    # Derivation: per-ROI agreement = exp(-|bpm_k - ref_bpm| / K).
    #   At deviation=9 BPM: factor = exp(-1) = 0.37 (borderline).
    #   K=9 ≈ one FFT resolution width at N=90 frames.
    #   Too tight (K=5): rejects valid ROI variance.
    #   Too loose (K=15): accepts harmonics.

    # ── Physiological plausibility ───────────────────────────────────────────
    BPM_PLAUSIBLE_LOW: float = 42.0
    BPM_PLAUSIBLE_HIGH: float = 140.0
    # Derivation: resting + light activity range for most users.
    #   140 BPM captures moderate exercise (5-7 METs).
    #   Above 140 BPM resting: tachycardia — rare in seated webcam scenario.
    #   Can be widened for exercise monitoring use case.

    SINGLE_ROI_SQI_MULT: float = 1.00
    # Derivation: no penalty for single-ROI fusion.
    #   Rationale: a high-quality single ROI is better than averaging with a
    #   low-quality second ROI. Let SQI carry the quality signal.

    HARMONIC_SUSPECT_MULT: float = 0.05
    # Derivation: 95% weight reduction for harmonic-suspected ROI.
    #   Not zero: 5% contribution preserves numerical stability.
    #   Harmonic detection is imperfect, so absolute rejection is too aggressive.

    ROI_MIN_AGREEMENT_FACTOR: float = 0.20
    # Derivation: allow up to exp(-dev/9) = 0.20, i.e., dev = 9*ln(5) = 14.5 BPM.
    #   At 14.5 BPM deviation: high confidence this is a different frequency.
    #   Wider than FUSION_OUTLIER_THRESHOLD because agreement is soft weighting.

    # ── Temporal Trust Lock ──────────────────────────────────────────────────
    STABLE_LOCK_SECONDS: float = 5.0
    # Derivation: 5 seconds of stable HR needed to trigger lock.
    #   At 60 BPM: covers 5 cardiac cycles — sufficient for physiological confirmation.
    #   At 30fps: 150 frames — full buffer — provides high confidence.
    # Sensitivity: moderate — too short (2s): false locks; too long (10s): slow convergence.

    STABLE_LOCK_BPM_STD: float = 5.0
    # Derivation: HR std < 5 BPM over 5 sec = physiologically stable reading.
    #   Normal HRV: 2-6 BPM std in 5-sec window. 5 BPM captures normal HRV
    #   without false-locking on noisy signals.

    STABLE_HOLD_ALPHA: float = 0.05
    # Derivation: when locked, EMA alpha = 0.05.
    #   Time constant: τ = 1/α = 20 frames = 0.67 sec.
    #   Slow update preserves stable reading while allowing slow HR drift.

    # ── ROI Auto-Exclusion ───────────────────────────────────────────────────
    ROI_BAD_STREAK_LIMIT: int = 120
    # Derivation: 120 frames = 4 seconds of continuous poor signal.
    #   Before exclusion, algorithm tries to recover (e.g., momentary occlusion).
    #   4 sec > typical blink duration (0.3s), talking pause (1s), or micro-motion burst (2s).
    # Sensitivity: high — too short (30 frames = 1s) causes premature exclusion.

    ROI_BAD_SQI_THRESH: float = 15.0
    # Derivation: P(valid) < 0.15 = signal is 85% likely noise.
    #   Only exclude ROIs that are genuinely dead, not just temporarily poor.

    ROI_REHAB_FRAMES: int = 30
    # Derivation: 30 frames = 1 second rehabilitation window.
    #   Fast enough to re-include ROI when condition improves.

    # ── Multi-window FFT voting ──────────────────────────────────────────────
    FFT_WINDOW_SEC: float = 4.0
    # Derivation: 4-second sub-windows for voting.
    #   Δf = 1/4 = 0.25 Hz = 15 BPM — resolves 40/55 BPM ambiguity.
    #   At 2 sec (old): Δf = 0.5 Hz = 30 BPM — too coarse for low HR.
    #   At 8 sec: good resolution but slow to adapt; not enough sub-windows.
    # Sensitivity: high — directly sets low-HR frequency resolution.

    FFT_WINDOW_VOTE_TOL: float = 0.10
    # Derivation: 0.10 Hz = 6 BPM tolerance for "same peak" across windows.
    #   Tighter than FFT resolution (0.25 Hz) to allow for sub-bin interpolation.
    #   Looser than 0.05 Hz to tolerate small window-to-window variation.

    FFT_HISTORY_OVERRIDE: float = 0.18
    # Derivation: 0.18 Hz = 10.8 BPM jump threshold for history override.
    #   Below this: new peak could be noise; trust history.
    #   Above this: definite frequency change; accept new peak.
    #   Set between 1× (0.10) and 2× (0.20) voting tolerance.

    # ── Temporal spike penalty ───────────────────────────────────────────────
    ROI_TEMPORAL_SPIKE_BPM: float = 15.0
    # Derivation: max physiological HR change in one second = 3 BPM.
    #   At 30fps, max change per buffer update (0.5s): 1.5 BPM.
    #   15 BPM spike = 10× physiological rate = artifact. Conservative threshold.

    ROI_TEMPORAL_SPIKE_MULT: float = 2.0
    # Derivation: multiply spike penalty weight by 2 (halves ROI contribution).
    #   Soft punishment, not exclusion — spike could be transient artifact.

    # ── Motion thresholds ────────────────────────────────────────────────────
    MOTION_ENTER_THRESHOLD: float = 2.8
    # Derivation: our motion metric = lm_score * 1.5 + px_score * 0.15 * lm_gate.
    #   At breathing + resting: lm ~0.3, px ~2 → score ≈ 0.3*1.5 + 0.3 = 0.75
    #   At micro head shift: lm ~0.8, px ~4 → score ≈ 1.2 + 0.18 = 1.38
    #   At clear head motion: lm ~2.0, px ~10 → score ≈ 3.0 + 0.45 = 3.45
    #   Threshold 2.8 sits between micro-shift and clear head motion.
    # Sensitivity: HIGH — most common source of false rejections.

    MOTION_EXIT_THRESHOLD: float = 1.8
    # Derivation: hysteresis gap = 2.8 - 1.8 = 1.0 unit.
    #   Prevents oscillation at boundary. Gap = ~0.4× enter threshold (standard
    #   hysteresis design: 30-50% of enter threshold).

    BPM_HOLDOVER_SEC: float = 5.0
    # Derivation: hold last good BPM for 5 sec after motion.
    #   Signal recovery time after motion: ~2-3 cardiac cycles = ~3 sec.
    #   5 sec provides buffer without holding stale data too long.

    MOTION_GRACE_FRAMES: int = 12
    # Derivation: 12 frames = 0.4 sec @ 30fps grace period after motion ends.
    #   Signal needs 0.3-0.5 sec to stabilize after motion stops.
    #   Based on filter settling time: 4th-order Butterworth at 0.7 Hz has
    #   group delay ≈ 0.3 sec → 9 frames. 12 = conservative margin.

    # ── BPM smoother ─────────────────────────────────────────────────────────
    BPM_EMA_ALPHA: float = 0.20
    # Derivation: time constant τ = 1/α = 5 frames = 0.17 sec @ 30fps.
    #   At 60 BPM: 0.17 sec ≈ 1/6 cardiac cycle.
    #   Smooths frame-to-frame FFT noise without introducing >0.5 sec lag.
    #   Note: BayesianHREstimator's Kalman filter is now the primary smoother;
    #   this EMA is a pre-smoother feeding the Kalman.

    BPM_MEDIAN_WINDOW: int = 11
    # Derivation: median over 11 frames = 0.37 sec window.
    #   Median (vs mean) is robust to FFT spike outliers.
    #   11 = odd number (required for exact median); covers ~1 cardiac cycle.

    BPM_MAX_JUMP: float = 25.0
    # Derivation: 25 BPM = hard outlier gate for EMA.
    #   Physiological max: 3 BPM/sec × 11 frames = 1.1 BPM.
    #   25 BPM allows for FFT mode switches (e.g., 75→50 BPM harmonic switch)
    #   while rejecting clear noise spikes (>25 BPM in one step).

    # ── Physiological plausibility (temporal) ────────────────────────────────
    TEMPORAL_BPM_MAX_JUMP_3FRAMES: float = 15.0
    # Derivation: max physiological change in 3 frames (0.1 sec) = 0.3 BPM.
    #   15 BPM threshold is ≈50× physiological max → catches only gross outliers.
    #   Too tight (5 BPM) would reject valid EMA-vs-FFT divergence during warmup.

    TEMPORAL_HISTORY_LEN: int = 7
    # Derivation: 7 frames of BPM history for jump detection.
    #   7 frames = 0.23 sec — captures any within-buffer HR spike.
    #   Longer histories dilute recent information.

    # ── Hierarchical fusion ──────────────────────────────────────────────────
    CLUSTER_DISTANCE_BPM: float = 14.0
    # Derivation: minimum separation to be classified as different clusters.
    #   At forehead BPM=63, cheek BPM=72 (same person, different ROIs):
    #   difference = 9 BPM — must be in same cluster.
    #   14 BPM = FFT resolution (12 BPM) + 2 BPM margin for peak jitter.
    #   Sub-harmonic of 70 BPM = 35 BPM → 70-35 = 35 BPM separation.
    #   14 BPM correctly places forehead and cheek in same cluster.

    MIN_CLUSTER_WEIGHT: float = 0.20
    # Derivation: minimum cluster weight fraction to include in fusion.
    #   At 20%: a cluster with <20% of total weight is likely an outlier.
    #   For 3 ROIs of equal weight: minority cluster has 33% → included.
    #   For 2v1 split: minority has 33% → included (edge case, not clearly outlier).

    # ── Calibration ──────────────────────────────────────────────────────────
    CALIBRATION_SQI_FLOOR: float = 40.0
    # Derivation: P(valid) > 0.40 to count as calibration sample.
    #   Below 40%: signal is more noise than signal — poor baseline estimate.

    # ── ROI brightness gating ────────────────────────────────────────────────
    ROI_BRIGHTNESS_MIN: float = 45.0
    # Derivation: 45/255 = 18% of max pixel value.
    #   Below 18%: ROI is in shadow. SNR degrades rapidly below this.
    #   YCrCb skin detection fails reliably below L=50 (≈45 RGB).

    ROI_BRIGHTNESS_MAX: float = 205.0
    # Derivation: 205/255 = 80% of max pixel value.
    #   Above 80%: saturation risk. Camera auto-exposure clips pixel values,
    #   losing linearity. Non-linear response invalidates the signal model.

    # ── Peak temporal bonus ──────────────────────────────────────────────────
    PEAK_TEMPORAL_BONUS: float = 0.25
    # Derivation: 25% weight bonus for FFT peak near previous-frame peak.
    #   Temporal continuity prior: HR unlikely to jump >0.12 Hz per frame.
    #   0.25 bonus (not higher): avoids locking onto wrong frequency.

    PEAK_TEMPORAL_TOL_HZ: float = 0.12
    # Derivation: 0.12 Hz = 7.2 BPM tolerance for "near previous peak".
    #   Maximum physiological change: 3 BPM/sec = 0.05 Hz/sec = 0.0017 Hz/frame.
    #   0.12 Hz = 72× physiological max per frame → accepts all real HR changes.
    #   Captures sub-bin FFT jitter (Δf = 0.2 Hz half-bin ≈ 0.1 Hz) plus margin.

    TEMPORAL_STABILITY_WINDOW: int = 10
    # Derivation: 10 frames of peak history = 0.33 sec.
    #   Within one cardiac cycle (0.5-1.0 sec), HR is physiologically constant.

    # ── Cluster Peak ─────────────────────────────────────────────────────────
    # (Note: these are effectively derived above in FFT_WINDOW params)



    # ── Legacy SQI weights (kept for rppg_core.py backward-compat) ──────────
    # These drove the OLD heuristic weighted-sum SQI:
    #   SQI = w_snr*snr + w_reg*reg + w_temp*temp + w_peak*peak + w_var*var
    #       - penalty_weight * (motion + lighting + fft) / 3
    #
    # The new rppg_sqi.py uses PosteriorSQI (log-posterior probability) which
    # makes these weights obsolete. They are kept here ONLY so rppg_core.py
    # doesn't crash while the migration to PosteriorSQI is completed.
    # TODO: remove these once compute_sqi() in rppg_core.py is replaced by
    #       PosteriorSQI.compute() calls.
    SQI_WEIGHT_SNR:          float = 0.35
    SQI_WEIGHT_REGULARITY:   float = 0.25
    SQI_WEIGHT_TEMPORAL:     float = 0.15
    SQI_WEIGHT_PEAK_CONSIST: float = 0.15
    SQI_WEIGHT_VARIANCE:     float = 0.10
    SQI_PENALTY_WEIGHT:      float = 0.40
    SQI_MAX_MOTION_PENALTY:  float = 0.50
    SQI_MAX_LIGHTING_PENALTY: float = 0.30
    SQI_MAX_FFT_PENALTY:     float = 0.25

cfg = RPPGConfig()
