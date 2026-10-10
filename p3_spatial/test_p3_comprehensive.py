"""
P3 Comprehensive Test Suite

Runs all P3 stage tests and reports results.

Test categories:
- P3.1: Spatial Quality Map
- P3.2: Motion Field
- P3.3: Motion-Aware Quality
- P3.4: Dynamic Spatial Candidates
- P3.5: Spatial-Temporal Consistency
- P3.6: P3 Pipeline Integration
- P3.7: Failure Analysis

Usage:
    python -m p3_spatial.test_p3_comprehensive
"""

import sys
import traceback
from typing import Tuple, List, Dict, Any


def run_module_tests(module_name: str, test_functions: List[Tuple[str, callable]]) -> Dict[str, Any]:
    """Run tests for a single module."""
    results = {
        "module": module_name,
        "passed": 0,
        "failed": 0,
        "errors": 0,
        "details": []
    }

    for name, fn in test_functions:
        try:
            if fn():
                results["passed"] += 1
                results["details"].append((name, "PASS", None))
            else:
                results["failed"] += 1
                results["details"].append((name, "FAIL", "Test returned False"))
        except AssertionError as e:
            results["failed"] += 1
            results["details"].append((name, "FAIL", str(e)))
        except Exception as e:
            results["errors"] += 1
            results["details"].append((name, "ERROR", f"{type(e).__name__}: {e}"))

    return results


def run_all_p3_tests() -> Dict[str, Any]:
    """Run all P3 tests across all stages."""

    print("=" * 80)
    print("P3 SPATIAL/MOTION EVOLUTION - COMPREHENSIVE TEST SUITE")
    print("=" * 80)
    print()

    all_results = []
    total_passed = 0
    total_failed = 0
    total_errors = 0

    # P3.1: Spatial Quality Map
    print("Loading P3.1: Spatial Quality Map...")
    try:
        from p3_spatial.spatial_quality_map import (
            test_uniform_quality,
            test_high_vs_low_quality,
            test_multiple_cardiac_frequencies,
            test_invalid_inputs,
            test_deterministic,
            test_zero_variance_signal
        )

        p3_1_tests = [
            ("uniform_quality", test_uniform_quality),
            ("high_vs_low_quality", test_high_vs_low_quality),
            ("multiple_cardiac_frequencies", test_multiple_cardiac_frequencies),
            ("invalid_inputs", test_invalid_inputs),
            ("deterministic", test_deterministic),
            ("zero_variance_signal", test_zero_variance_signal),
        ]

        results = run_module_tests("P3.1 Spatial Quality Map", p3_1_tests)
        all_results.append(results)
        total_passed += results["passed"]
        total_failed += results["failed"]
        total_errors += results["errors"]
    except ImportError as e:
        print(f"  ERROR: Cannot import P3.1: {e}")
        total_errors += 1

    # P3.2: Motion Field
    print("Loading P3.2: Motion Field...")
    try:
        from p3_spatial.motion_field import (
            test_static_face,
            test_uniform_motion,
            test_localized_motion,
            test_unstable_motion,
            test_deterministic
        )

        p3_2_tests = [
            ("static_face", test_static_face),
            ("uniform_motion", test_uniform_motion),
            ("localized_motion", test_localized_motion),
            ("unstable_motion", test_unstable_motion),
            ("deterministic", test_deterministic),
        ]

        results = run_module_tests("P3.2 Motion Field", p3_2_tests)
        all_results.append(results)
        total_passed += results["passed"]
        total_failed += results["failed"]
        total_errors += results["errors"]
    except ImportError as e:
        print(f"  ERROR: Cannot import P3.2: {e}")
        total_errors += 1

    # P3.3: Motion-Aware Quality
    print("Loading P3.3: Motion-Aware Quality...")
    try:
        from p3_spatial.motion_aware_quality import (
            test_quality_only_mode,
            test_motion_only_mode,
            test_full_mode,
            test_high_motion_downweighting,
            test_clean_region_preserved,
            test_shape_mismatch,
            test_empty_input
        )

        p3_3_tests = [
            ("quality_only_mode", test_quality_only_mode),
            ("motion_only_mode", test_motion_only_mode),
            ("full_mode", test_full_mode),
            ("high_motion_downweighting", test_high_motion_downweighting),
            ("clean_region_preserved", test_clean_region_preserved),
            ("shape_mismatch", test_shape_mismatch),
            ("empty_input", test_empty_input),
        ]

        results = run_module_tests("P3.3 Motion-Aware Quality", p3_3_tests)
        all_results.append(results)
        total_passed += results["passed"]
        total_failed += results["failed"]
        total_errors += results["errors"]
    except ImportError as e:
        print(f"  ERROR: Cannot import P3.3: {e}")
        total_errors += 1

    # P3.4: Dynamic Spatial Candidates
    print("Loading P3.4: Dynamic Spatial Candidates...")
    try:
        from p3_spatial.dynamic_candidates import (
            test_no_candidates_below_threshold,
            test_single_candidate,
            test_multiple_candidates,
            test_candidate_persistence,
            test_motion_contamination_rejection,
            test_max_candidates_limit,
            test_empty_quality_map
        )

        p3_4_tests = [
            ("no_candidates_below_threshold", test_no_candidates_below_threshold),
            ("single_candidate", test_single_candidate),
            ("multiple_candidates", test_multiple_candidates),
            ("candidate_persistence", test_candidate_persistence),
            ("motion_contamination_rejection", test_motion_contamination_rejection),
            ("max_candidates_limit", test_max_candidates_limit),
            ("empty_quality_map", test_empty_quality_map),
        ]

        results = run_module_tests("P3.4 Dynamic Spatial Candidates", p3_4_tests)
        all_results.append(results)
        total_passed += results["passed"]
        total_failed += results["failed"]
        total_errors += results["errors"]
    except ImportError as e:
        print(f"  ERROR: Cannot import P3.4: {e}")
        total_errors += 1

    # P3.5: Spatial-Temporal Consistency
    print("Loading P3.5: Spatial-Temporal Consistency...")
    try:
        from p3_spatial.spatial_temporal import (
            test_quality_stability_computation,
            test_persistence_score,
            test_candidate_birth,
            test_candidate_degradation,
            test_candidate_expiration,
            test_recovery,
            test_global_stability
        )

        p3_5_tests = [
            ("quality_stability_computation", test_quality_stability_computation),
            ("persistence_score", test_persistence_score),
            ("candidate_birth", test_candidate_birth),
            ("candidate_degradation", test_candidate_degradation),
            ("candidate_expiration", test_candidate_expiration),
            ("recovery", test_recovery),
            ("global_stability", test_global_stability),
        ]

        results = run_module_tests("P3.5 Spatial-Temporal Consistency", p3_5_tests)
        all_results.append(results)
        total_passed += results["passed"]
        total_failed += results["failed"]
        total_errors += results["errors"]
    except ImportError as e:
        print(f"  ERROR: Cannot import P3.5: {e}")
        total_errors += 1

    # P3.6: P3 Pipeline Integration
    print("Loading P3.6: P3 Pipeline Integration...")
    try:
        from p3_spatial.p3_pipeline import (
            test_pipeline_initialization,
            test_e0_baseline,
            test_e1_quality_only,
            test_e6_full_pipeline,
            test_buffer_processing,
            test_ablation_metrics,
            test_reset
        )

        p3_6_tests = [
            ("pipeline_initialization", test_pipeline_initialization),
            ("e0_baseline", test_e0_baseline),
            ("e1_quality_only", test_e1_quality_only),
            ("e6_full_pipeline", test_e6_full_pipeline),
            ("buffer_processing", test_buffer_processing),
            ("ablation_metrics", test_ablation_metrics),
            ("reset", test_reset),
        ]

        results = run_module_tests("P3.6 P3 Pipeline Integration", p3_6_tests)
        all_results.append(results)
        total_passed += results["passed"]
        total_failed += results["failed"]
        total_errors += results["errors"]
    except ImportError as e:
        print(f"  ERROR: Cannot import P3.6: {e}")
        total_errors += 1

    # P3.7: Failure Analysis
    print("Loading P3.7: Failure Analysis...")
    try:
        from p3_spatial.p3_failures import (
            test_motion_failure_detection,
            test_low_motion_no_failure,
            test_insufficient_candidates,
            test_temporal_instability,
            test_spatial_disagreement,
            test_candidate_thrashing,
            test_failure_response,
            test_reset
        )

        p3_7_tests = [
            ("motion_failure_detection", test_motion_failure_detection),
            ("low_motion_no_failure", test_low_motion_no_failure),
            ("insufficient_candidates", test_insufficient_candidates),
            ("temporal_instability", test_temporal_instability),
            ("spatial_disagreement", test_spatial_disagreement),
            ("candidate_thrashing", test_candidate_thrashing),
            ("failure_response", test_failure_response),
            ("reset", test_reset),
        ]

        results = run_module_tests("P3.7 Failure Analysis", p3_7_tests)
        all_results.append(results)
        total_passed += results["passed"]
        total_failed += results["failed"]
        total_errors += results["errors"]
    except ImportError as e:
        print(f"  ERROR: Cannot import P3.7: {e}")
        total_errors += 1

    # Print summary
    print()
    print("=" * 80)
    print("P3 COMPREHENSIVE TEST SUMMARY")
    print("=" * 80)
    print()

    n_tests = 0
    for r in all_results:
        n = r["passed"] + r["failed"] + r["errors"]
        n_tests += n
        status = "OK" if r["failed"] == 0 and r["errors"] == 0 else "FAIL"
        print(f"  [{status}] {r['module']}: {r['passed']}/{n} passed", end="")
        if r["failed"] > 0:
            print(f", {r['failed']} failed", end="")
        if r["errors"] > 0:
            print(f", {r['errors']} errors", end="")
        print()

    print()
    print("-" * 80)
    print(f"TOTAL: {total_passed}/{n_tests} passed", end="")
    if total_failed > 0:
        print(f", {total_failed} failed", end="")
    if total_errors > 0:
        print(f", {total_errors} errors", end="")
    print()
    print("=" * 80)

    return {
        "results": all_results,
        "total_passed": total_passed,
        "total_failed": total_failed,
        "total_errors": total_errors,
        "total_tests": n_tests
    }


def main():
    """Run all P3 tests and exit with appropriate code."""
    try:
        results = run_all_p3_tests()

        if results["total_failed"] > 0 or results["total_errors"] > 0:
            print("\nWARNING: Some tests failed. Review output above.")
            return 1
        else:
            print("\nSUCCESS: All P3 tests passed.")
            return 0
    except Exception as e:
        print(f"\nERROR: Fatal error running tests: {type(e).__name__}: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
