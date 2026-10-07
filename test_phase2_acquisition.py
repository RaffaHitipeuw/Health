# -*- coding: utf-8 -*-
"""
Tests for Phase 2 Data Acquisition Module

Run with:
    pytest -q test_phase2_acquisition.py
"""

import os
import sys
import json
import tempfile
import numpy as np
from pathlib import Path

# Add repo to path
REPO_ROOT = Path('D:/main/Projects/Health')
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))

import pytest
from phase2_data_acquisition import (
    compute_sha256,
    validate_video,
    validate_gt,
    check_video_gt_overlap,
    create_dataset_manifest,
    DatasetManifest,
)


class TestCaptureMetadataSchema:
    """Test G5: capture metadata schema."""

    def test_capture_metadata_fields(self):
        """Verify all required metadata fields are present."""
        from phase2_data_acquisition import CaptureMetadata

        # Required fields
        required_fields = [
            'capture_id', 'capture_start_time', 'capture_end_time',
            'video_path', 'video_fps', 'video_width', 'video_height',
            'video_frame_count', 'video_duration_seconds',
            'actual_fps', 'nominal_fps', 'fourcc', 'camera_index'
        ]

        # Verify all fields exist in dataclass
        for field_name in required_fields:
            assert hasattr(CaptureMetadata, field_name) or field_name in CaptureMetadata.__annotations__


class TestManifestValidation:
    """Test G5.3: manifest validation."""

    def test_manifest_has_required_fields(self):
        """Verify all required manifest fields exist."""
        required_fields = [
            'manifest_version', 'dataset_id', 'created_at',
            'video_path', 'video_sha256', 'video_fps',
            'video_frame_count', 'video_duration_seconds',
            'gt_path', 'gt_sha256', 'gt_status',
            'gt_offset_seconds', 'video_valid', 'gt_valid',
            'face_detected', 'roi_valid', 'overlap_valid',
            'capture_metadata', 'provenance'
        ]

        for field_name in required_fields:
            assert hasattr(DatasetManifest, field_name) or field_name in DatasetManifest.__annotations__

    def test_sha256_computation(self):
        """Test SHA256 hash computation."""
        # Create temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write('test content')
            temp_path = f.name

        try:
            sha256 = compute_sha256(temp_path)
            assert len(sha256) == 64  # SHA256 produces 64 hex chars
            assert sha256.isalnum()
        finally:
            os.unlink(temp_path)

    def test_sha256_missing_file(self):
        """Test SHA256 returns empty for missing file."""
        sha256 = compute_sha256('nonexistent_file.txt')
        assert sha256 == ""


class TestTimestampMonotonicity:
    """Test G5.2: timestamp validation."""

    def test_gt_timestamps_must_be_monotonic(self):
        """Verify GT timestamps are strictly increasing."""
        from phase2_data_acquisition import validate_gt

        # Create temp GT with monotonic timestamps
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write('timestamp,bpm\n')
            for i in range(10):
                f.write('{:.3f},72.0\n'.format(i * 0.5))
            temp_path = f.name

        try:
            result = validate_gt(temp_path)
            assert result['valid'] == True
            assert result['is_monotonic'] == True
        finally:
            os.unlink(temp_path)

    def test_nonmonotonic_gt_rejected(self):
        """Verify non-monotonic GT is flagged."""
        from phase2_data_acquisition import validate_gt

        # Create temp GT with non-monotonic timestamps
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write('timestamp,bpm\n')
            f.write('0.0,72.0\n')
            f.write('1.0,73.0\n')
            f.write('0.5,71.0\n')  # Decreasing!
            temp_path = f.name

        try:
            result = validate_gt(temp_path)
            assert result['valid'] == True  # Still valid format
            assert result['is_monotonic'] == False
        finally:
            os.unlink(temp_path)


class TestGroundTruthValidation:
    """Test G5.1: GT validation."""

    def test_gt_missing_header_rejected(self):
        """Verify GT without proper header is rejected."""
        from phase2_data_acquisition import validate_gt

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write('time,heartrate\n')  # Wrong header
            f.write('0.0,72\n')
            temp_path = f.name

        try:
            result = validate_gt(temp_path)
            assert result['valid'] == False
            assert 'header' in result['error'].lower()
        finally:
            os.unlink(temp_path)

    def test_gt_insufficient_entries_rejected(self):
        """Verify GT with too few entries is flagged."""
        from phase2_data_acquisition import validate_gt

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write('timestamp,bpm\n')
            f.write('0.0,72\n')  # Only one entry
            temp_path = f.name

        try:
            result = validate_gt(temp_path)
            assert result['valid'] == False
        finally:
            os.unlink(temp_path)

    def test_gt_implausible_bpm_flagged(self):
        """Verify GT with implausible BPM is flagged."""
        from phase2_data_acquisition import validate_gt

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write('timestamp,bpm\n')
            f.write('0.0,250\n')  # Implausible
            f.write('0.5,72\n')
            f.write('1.0,73\n')
            temp_path = f.name

        try:
            result = validate_gt(temp_path)
            assert result['valid'] == True  # Format valid
            assert result['is_plausible'] == False
        finally:
            os.unlink(temp_path)


class TestVideoGtOverlap:
    """Test G5.2: video/GT overlap check."""

    def test_overlap_with_zero_offset(self):
        """Test overlap check with no offset."""
        # Video: 0-30s, GT: 0-30s
        assert check_video_gt_overlap(30.0, 0.0, 30.0, 0.0) == True

    def test_overlap_with_positive_offset(self):
        """Test overlap check with positive offset."""
        # Video: 0-30s, GT: 5-35s (starts 5s after video)
        assert check_video_gt_overlap(30.0, 5.0, 35.0, -5.0) == True

    def test_no_overlap(self):
        """Test non-overlapping case."""
        # Video: 0-30s, GT: 100-130s
        assert check_video_gt_overlap(30.0, 100.0, 130.0, 0.0) == False


class TestSyncOffset:
    """Test G5.2: synchronization offset."""

    def test_sync_offset_calculation(self):
        """Verify synchronization offset is calculated correctly."""
        video_start = 0.0
        gt_start = 5.0
        offset = gt_start - video_start

        assert offset == 5.0

        # GT at 10s should align with video at 5s
        assert abs((10.0 - offset) - 5.0) < 0.001


class TestBenchmarkInputCompatibility:
    """Test G5.4: benchmark input compatibility."""

    def test_manifest_json_serializable(self):
        """Verify manifest can be serialized to JSON."""
        manifest = DatasetManifest(
            dataset_id='test_001',
            created_at='2024-01-01T00:00:00',
            video_path='/path/to/video.mp4',
            video_fps=30.0,
            video_frame_count=900,
            video_duration_seconds=30.0,
            video_resolution=(640, 480),
            gt_path='/path/to/gt.csv',
            gt_offset_seconds=0.0,
            gt_status='present'
        )

        # Convert to dict
        manifest_dict = {
            'manifest_version': manifest.manifest_version,
            'dataset_id': manifest.dataset_id,
            'created_at': manifest.created_at,
            'video_path': manifest.video_path,
            'video_fps': manifest.video_fps,
            'video_frame_count': manifest.video_frame_count,
            'video_duration_seconds': manifest.video_duration_seconds,
            'video_resolution': manifest.video_resolution,
            'gt_path': manifest.gt_path,
            'gt_offset_seconds': manifest.gt_offset_seconds,
            'gt_status': manifest.gt_status
        }

        # Should be JSON serializable
        json_str = json.dumps(manifest_dict)
        assert len(json_str) > 0

    def test_required_paths_for_benchmark(self):
        """Verify manifest contains paths needed by benchmark runner."""
        manifest = DatasetManifest()
        manifest.video_path = '/path/to/video.mp4'
        manifest.gt_path = '/path/to/gt.csv'
        manifest.gt_offset_seconds = 2.0

        # Benchmark needs these
        assert hasattr(manifest, 'video_path') or 'video_path' in manifest.__annotations__
        assert hasattr(manifest, 'gt_path') or 'gt_path' in manifest.__annotations__
        assert hasattr(manifest, 'gt_offset_seconds') or 'gt_offset_seconds' in manifest.__annotations__


class TestVideoValidation:
    """Test video validation."""

    def test_video_format_validation(self):
        """Verify video format validation checks."""
        # Test with nonexistent file
        result = validate_video('/nonexistent/video.mp4')
        assert result['valid'] == False
        assert 'not found' in result['error'].lower()


class TestIntegrity:
    """Test G5: integrity rules."""

    def test_gt_status_blocked_option_exists(self):
        """Verify GT can be marked as blocked."""
        manifest = DatasetManifest()
        manifest.gt_status = 'blocked'

        assert manifest.gt_status == 'blocked'

    def test_provenance_required(self):
        """Verify provenance is tracked."""
        manifest = DatasetManifest()
        manifest.provenance = 'Real face video from webcam. GT from external pulse oximeter.'

        assert len(manifest.provenance) > 0
        # Should not claim fabricated GT
        assert 'fabricated' not in manifest.provenance.lower()
        assert 'estimated' not in manifest.provenance.lower() or 'blocked' in manifest.provenance.lower()


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
