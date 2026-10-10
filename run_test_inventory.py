#!/usr/bin/env python3
"""Comprehensive test inventory for P2 pre-execution audit."""

import sys
import importlib

print('=' * 70)
print('COMPREHENSIVE TEST INVENTORY')
print('=' * 70)
print()

# Test files to run
test_files = [
    'test_benchmark_infrastructure',
    'test_benchmark_alignment',
    'test_ground_truth',
]

results = []
total_passed = 0
total_failed = 0

for module_name in test_files:
    try:
        module = importlib.import_module(module_name)
        if hasattr(module, 'run_tests'):
            result = module.run_tests()
            results.append({
                'module': module_name,
                'passed': result,
                'failed': not result,
                'status': 'PASS' if result else 'FAIL'
            })
            if result:
                total_passed += 1
            else:
                total_failed += 1
        else:
            results.append({
                'module': module_name,
                'passed': False,
                'failed': True,
                'status': 'NO run_tests()'
            })
            total_failed += 1
    except Exception as e:
        results.append({
            'module': module_name,
            'passed': False,
            'failed': True,
            'status': f'ERROR: {e}'
        })
        total_failed += 1

print('TEST RESULTS:')
print('-' * 70)
for r in results:
    status = r['status']
    print(f"  {r['module']}: {status}")

print()
print('=' * 70)
print(f'MODULES: {total_passed} passed, {total_failed} failed')
print('=' * 70)

sys.exit(0 if total_failed == 0 else 1)
