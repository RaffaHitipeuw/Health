# rPPG System Upgrade Summary - May 2026

## 1. Harmonic Rejection & Physiological Gating (PRIORITY)
- **Harmonic Rejection Layer**: Diimplementasikan di `_pick_one_peak` untuk mendeteksi harmonic kedua (2x BPM).
- **Penalty Logic**: Menambahkan penalti kepercayaan 0.4x jika kandidat BPM mendekati 2x dari BPM sebelumnya (`abs(bpm_candidate - bpm_prev * 2) < 8`).
- **Physiological Gating**: Jika kandidat BPM > 130 dan terdapat peak lain di sekitar setengah frekuensinya, kandidat tersebut akan di-reject (mengasumsikan itu adalah harmonic).

## 2. Temporal Consistency (Historical Inertia)
- **Historical Inertia**: Mengganti logika fusion yang terlalu reaktif dengan formula inertia: `final_bpm = current_bpm * 0.3 + prev_bpm * 0.7`.
- **Extreme Jump Protection**: Menambahkan proteksi ekstra untuk lompatan BPM yang ekstrem (> MAX_BPM_JUMP) dengan memberikan inertia yang lebih berat (0.9x prev_bpm).

## 3. Robust Agreement & ROI Trust Fusion
- **Agreement Metric**: Sekarang menggabungkan kedekatan BPM dan validitas fisiologis (historical validity).
- **Adaptive ROI Trust Memory**: Mengimplementasikan logika "Trust strongest physiologically plausible ROI" dengan mengurutkan ROI berdasarkan skor trust (learned_weight * SQI) sebelum melakukan fusion.
- **Dominant ROI Boosting**: Jika satu ROI memiliki skor trust yang jauh lebih tinggi (1.5x) dari yang lain, ROI tersebut akan diberikan bobot lebih dominan untuk menghindari "salah bersama".

## 4. Signal Quality & REG Metric Improvements
- **Improved Detrending**: Menggabungkan linear detrend dengan moving average subtraction (window 2 detik) untuk menangani non-linear DC drift atau illumination drift.
- **Improved Bandpass**: Menggunakan SOS (Second-Order Sections) untuk stabilitas numerik yang lebih baik.
- **Peak Dominance Ratio**: Menambahkan penalti pada log-posterior jika peak utama tidak cukup dominan dibandingkan peak lainnya (spectral entropy proxy).

## 5. Forehead ROI Adjustment
- **Forehead ROI Resizing**: Mengecilkan Forehead ROI dan menurunkannya mendekati alis (menghindari hairline). Menggunakan subset landmark yang lebih rendah: `[109, 67, 108, 151, 337, 297, 338]`.

## 6. Peak Metric Refinement
- **Harmonic Penalty**: Memberikan penalti pada SQI jika peak terdeteksi di area yang tidak manusiawi (>135 BPM) saat kondisi diam (motion < 1.0).
- **Spectral Entropy Proxy**: Menggunakan rasio dominansi peak untuk menurunkan confidence jika spektrum terlalu noisy.
