"""
Ground truth file loader for benchmark experiments.

Supports timestamped BPM ground truth in CSV format.

Format:
    timestamp,bpm
    0.000,72.0
    0.500,72.4
    1.000,71.8

Strict validation ensures scientific integrity.
"""

import csv
import os
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class GroundTruthEntry:
    """Single ground truth observation."""
    timestamp: float
    bpm: float


class GroundTruthLoadError(Exception):
    """Raised when GT file cannot be loaded or is malformed."""
    pass


@dataclass
class GroundTruthData:
    """
    Loaded ground truth data with validation metadata.
    """
    entries: List[GroundTruthEntry] = field(default_factory=list)
    source_file: str = ""
    validation_errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def timestamps(self) -> np.ndarray:
        """Array of ground truth timestamps."""
        return np.array([e.timestamp for e in self.entries])

    @property
    def bpms(self) -> np.ndarray:
        """Array of ground truth BPM values."""
        return np.array([e.bpm for e in self.entries])

    @property
    def n_entries(self) -> int:
        """Number of GT entries."""
        return len(self.entries)

    def is_valid(self) -> bool:
        """Check if data passed validation."""
        return len(self.validation_errors) == 0

    def to_series(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert to aligned series for metric computation.

        Returns (timestamps, bpms) as numpy arrays.
        """
        return self.timestamps, self.bpms


def load_ground_truth(
    gt_path: str,
    timestamp_col: str = "timestamp",
    bpm_col: str = "bpm",
) -> GroundTruthData:
    """
    Load ground truth from CSV file with strict validation.

    Parameters
    ----------
    gt_path : str
        Path to CSV ground truth file.
    timestamp_col : str
        Column name for timestamps. Default: "timestamp".
    bpm_col : str
        Column name for BPM values. Default: "bpm".

    Returns
    -------
    GroundTruthData
        Validated ground truth data.

    Raises
    ------
    GroundTruthLoadError
        If file cannot be read, has wrong format, or fails validation.

    Notes
    -----
    Validates:
    - File exists and is readable
    - Required columns present
    - Numeric timestamps
    - Numeric BPM values
    - Monotonic timestamps (strictly increasing)
    - No duplicate timestamps
    - Reasonable BPM range (20-250)
    - No missing values
    """
    if not os.path.exists(gt_path):
        raise GroundTruthLoadError(f"GT file not found: {gt_path}")

    data = GroundTruthData(source_file=gt_path)

    try:
        with open(gt_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            # Check required columns
            if reader.fieldnames is None:
                raise GroundTruthLoadError(f"Cannot read CSV headers from: {gt_path}")

            ts_col = None
            bpm_col_found = None

            for col in reader.fieldnames:
                col_lower = col.lower().strip()
                if col_lower == timestamp_col.lower():
                    ts_col = col
                if col_lower == bpm_col.lower():
                    bpm_col_found = col

            if ts_col is None:
                raise GroundTruthLoadError(
                    f"Timestamp column '{timestamp_col}' not found. "
                    f"Available columns: {list(reader.fieldnames)}"
                )
            if bpm_col_found is None:
                raise GroundTruthLoadError(
                    f"BPM column '{bpm_col}' not found. "
                    f"Available columns: {list(reader.fieldnames)}"
                )

            # Parse rows
            for row_idx, row in enumerate(reader, start=2):  # start=2 for header line
                row_num = row_idx + 1

                # Extract and validate timestamp
                ts_str = row.get(ts_col, "").strip()
                if not ts_str:
                    data.validation_errors.append(
                        f"Row {row_num}: Missing timestamp value"
                    )
                    continue

                try:
                    ts = float(ts_str)
                except ValueError:
                    data.validation_errors.append(
                        f"Row {row_num}: Non-numeric timestamp: '{ts_str}'"
                    )
                    continue

                # Extract and validate BPM
                bpm_str = row.get(bpm_col_found, "").strip()
                if not bpm_str:
                    data.validation_errors.append(
                        f"Row {row_num}: Missing BPM value at timestamp {ts}"
                    )
                    continue

                try:
                    bpm = float(bpm_str)
                except ValueError:
                    data.validation_errors.append(
                        f"Row {row_num}: Non-numeric BPM: '{bpm_str}' at timestamp {ts}"
                    )
                    continue

                # Validate BPM range (physiological bounds)
                if not (20.0 <= bpm <= 250.0):
                    data.warnings.append(
                        f"Row {row_num}: BPM {bpm} outside typical range [20, 250] at timestamp {ts}"
                    )

                data.entries.append(GroundTruthEntry(timestamp=ts, bpm=bpm))

    except csv.Error as e:
        raise GroundTruthLoadError(f"CSV parsing error: {e}")
    except UnicodeDecodeError:
        raise GroundTruthLoadError(f"File encoding error (expected UTF-8): {gt_path}")
    except GroundTruthLoadError:
        raise
    except Exception as e:
        raise GroundTruthLoadError(f"Unexpected error loading GT file: {e}")

    # Validation: Check for empty data
    if len(data.entries) == 0:
        data.validation_errors.append("No valid entries loaded from GT file")
        return data

    # Validation: Check monotonic timestamps
    timestamps = [e.timestamp for e in data.entries]
    for i in range(1, len(timestamps)):
        if timestamps[i] <= timestamps[i-1]:
            data.validation_errors.append(
                f"Timestamps not strictly increasing: "
                f"{timestamps[i-1]:.3f} -> {timestamps[i]:.3f} at index {i}"
            )

    # Validation: Check for duplicate timestamps
    seen_timestamps = set()
    for i, entry in enumerate(data.entries):
        if entry.timestamp in seen_timestamps:
            data.validation_errors.append(
                f"Duplicate timestamp {entry.timestamp} at entries {list(seen_timestamps).count(entry.timestamp)+1}"
            )
        seen_timestamps.add(entry.timestamp)

    return data


def load_ground_truth_for_interface(
    gt_path: str,
    source: str = "benchmark_gt",
) -> List[Tuple[float, float, str]]:
    """
    Load ground truth into the format expected by GroundTruthInterface.

    GroundTruthInterface expects: (timestamp, bpm, source) tuples.

    Parameters
    ----------
    gt_path : str
        Path to CSV ground truth file.
    source : str
        Source label for GroundTruthInterface.

    Returns
    -------
    List[Tuple[float, float, str]]
        List of (timestamp, bpm, source) tuples.

    Raises
    ------
    GroundTruthLoadError
        If loading or validation fails.
    """
    gt_data = load_ground_truth(gt_path)
    if not gt_data.is_valid():
        errors = "; ".join(gt_data.validation_errors[:5])
        raise GroundTruthLoadError(
            f"GT validation failed: {errors}"
        )
    return [(e.timestamp, e.bpm, source) for e in gt_data.entries]
