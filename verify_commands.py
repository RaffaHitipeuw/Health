#!/usr/bin/env python3
"""Command existence verification."""

import os
import sys

print('=' * 60)
print('COMMAND REPRODUCIBILITY AUDIT')
print('=' * 60)

# Check if command files exist
commands = [
    'benchmark_video.py',
    'phase2_data_acquisition.py',
    'rppg_phase2_realbenchmark.py',
    'rppg_benchmark_run.py',
    'benchmark_synthetic_example.py',
]

print()
print('CLI Commands:')
for cmd in commands:
    exists = os.path.exists(cmd)
    status = "EXISTS" if exists else "MISSING"
    print(f'  {cmd}: {status}')

print()
print('Benchmark infrastructure:')
bench_files = [
    'rppg_benchmark.py',
    'rppg_benchmark_gt.py',
    'rppg_benchmark_video.py',
    'rppg_benchmark_run.py',
    'rppg_benchmark_manifest.py',
    'rppg_benchmark_failures.py',
    'rppg_benchmark_alignment.py',
]
for f in bench_files:
    exists = os.path.exists(f)
    status = "EXISTS" if exists else "MISSING"
    print(f'  {f}: {status}')

print()
print('Phase 2 infrastructure:')
p2_files = [
    'phase2_data_acquisition.py',
    'rppg_phase2_signal_lab.py',
    'rppg_phase2_experiments.py',
    'rppg_phase2_evolution.py',
    'rppg_phase2_realbenchmark.py',
]
for f in p2_files:
    exists = os.path.exists(f)
    status = "EXISTS" if exists else "MISSING"
    print(f'  {f}: {status}')
