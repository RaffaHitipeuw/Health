# Technical Report: Advanced rPPG System Upgrades for Research-Grade Robustness

This report details the comprehensive technical enhancements implemented in the remote photoplethysmography (rPPG) system to address the rigorous requirements of high-fidelity physiological monitoring. The upgrades transition the system from a heuristic-heavy implementation to a statistically grounded, uncertainty-aware framework suitable for academic and clinical research.

## 1. Standardized Signal Extraction and Validation Framework

The core signal extraction pipeline has been expanded to include industry-standard algorithms, ensuring comparability with existing literature. The system now supports **Plane-Orthogonal-to-Skin (POS)**, **Chrominance-based (CHROM)**, **Independent Component Analysis (ICA)**, and **Green-channel averaging (GREEN)**. These are implemented in a modular fashion within `rppg_algorithms.py`, allowing for dynamic algorithm selection and benchmarking.

To address the requirement for ground-truth validation, the `GroundTruthInterface` has been significantly enhanced. It now supports rigorous statistical evaluation metrics, including **Bland-Altman analysis** for bias and limits of agreement, and **Pearson correlation coefficients** for linear relationship assessment. Furthermore, a `ReproducibilityProtocol` has been established to ensure that all experiments are documented with full configuration snapshots and environment metadata, facilitating repeatable research.

| Feature | Implementation | Purpose |
| :--- | :--- | :--- |
| **Standardized Algos** | POS, CHROM, ICA, GREEN | Benchmarking against reference methods |
| **Validation Metrics** | Bland-Altman, Pearson R, MAE, RMSE | Quantifying accuracy against ground truth |
| **Reproducibility** | Metadata logging & Config snapshots | Ensuring experimental repeatability |

## 2. Explicit Temporal Modeling and Probabilistic State Estimation

The system has moved beyond simple frame-window logic to explicit temporal modeling. A dual-stage approach has been implemented in `rppg_temporal.py`:

> **Kalman Filtering**: A state-space model tracks the BPM and its velocity, providing optimal estimates under Gaussian noise assumptions. This effectively dampens physiological impossibilities and handles short-term signal dropouts.
>
> **Bayesian Tracking**: A probabilistic grid-based tracker maintains a full probability distribution of the heart rate. This allows the system to represent multi-modal distributions and handle ambiguity when multiple periodic signals are present.

These models are integrated into an **Adaptive Probabilistic Fusion** engine. Instead of manual weighting, the system now fuses multiple Region of Interest (ROI) estimates by combining their likelihood functions, weighted by their respective uncertainty levels.

## 3. Statistically Calibrated Signal Quality and Uncertainty Decomposition

Signal Quality Index (SQI) calculation has been overhauled to be statistically calibrated. The new `StatisticallyCalibratedSQI` module utilizes **Signal-to-Noise Ratio (SNR)** in decibels and **Spectral Entropy** to quantify signal cleanliness. These metrics are normalized using sigmoid functions calibrated against baseline physiological distributions.

A critical addition is the **Uncertainty-Aware Session Confidence** module. It decomposes uncertainty into two distinct components:
1.  **Aleatoric Uncertainty**: Captures the inherent noise in the signal (e.g., sensor noise, lighting fluctuations).
2.  **Epistemic Uncertainty**: Reflects the model's lack of knowledge or data inconsistency (e.g., ROI disagreement, BPM variance).

This decomposition allows for the calculation of **95% Confidence Intervals**, providing a transparent measure of the estimate's reliability.

## 4. Spatial Robustness and Adaptive ROI Management

To handle cross-subject variability and motion, an `AdaptiveROIManager` has been implemented. This module utilizes robust **skin segmentation** in both YCrCb and HSV color spaces to ensure that only viable skin pixels contribute to the signal. 

The system now features **Adaptive Facial Perfusion Mapping**, which dynamically identifies areas with the strongest pulse signals within the segmented skin. This is complemented by **Learned Weighting**, where the system adaptively adjusts the importance of different ROIs (e.g., forehead vs. cheeks) based on their historical consensus performance during a session.

## 5. Signal Integrity and Interference Rejection

Advanced signal processing techniques have been integrated to mitigate common rPPG artifacts:
*   **FFT Anti-Leakage**: Implementation of Hanning windowing prior to spectral analysis to minimize spectral leakage and improve peak resolution.
*   **Mature Harmonic Rejection**: A heuristic-based detection system that identifies and rejects sub-harmonic and harmonic interference, preventing "double-counting" of the heart rate.
*   **Respiratory Interference Analysis**: Quantifies the ratio of respiratory-band power to pulse-band power, penalizing the SQI when respiratory artifacts dominate the signal.

## 6. Evaluation and Stress-Testing Suite

To validate these upgrades, an **Ablation Study Framework** has been developed. This allows researchers to systematically enable or disable components (e.g., "Kalman vs. No Kalman") to quantify their specific contribution to accuracy. Additionally, a **Stress-Test Benchmark** suite has been added to evaluate performance under challenging conditions, including low light, significant head rotation, and speaking.

---
*This upgraded codebase represents a significant leap in technical maturity, addressing the core concerns of rigorous physiological signal analysis.*
