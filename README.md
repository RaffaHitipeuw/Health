# Health

## Active Engine
The active rPPG processing engine is **MultiROIFusionEngineV2** (`rppg_core.py`).
Legacy baseline **MultiROIFusionEngine (V1)** is preserved but not used by default.

## Research System
This is a **research prototype** for camera-based physiological measurement.
- Core HR estimation from facial video is the primary validated component.
- Respiration rate, HRV metrics, stress index, arrhythmia detection, and AI interpretation are **EXPERIMENTAL**.
- This system is NOT a medical device and makes NO diagnostic claims.

## CLI Usage
```bash
python rppg_main.py --log  # Enable reproducibility logging
python rppg_main.py --help  # Show options
```

## Research Logging
Press `[L]` during a session to save reproducibility logs (CSV files) of raw signals, SQI breakdowns, and events.