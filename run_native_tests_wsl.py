#!/usr/bin/env python3
"""
run_native_tests_wsl.py — Compile and run all 18 C native tests using WSL gcc

Tests use Unity framework (tests/engine/unity/) for C testing.

Usage:
    python run_native_tests_wsl.py [--verbose] [--test-name <name>]

Returns:
    0 if all tests pass
    1 if any test fails
"""

import os
import sys
import subprocess
import json
from pathlib import Path
from typing import List, Dict, Tuple

PROJECT_ROOT = Path(__file__).parent
TESTS_DIR = PROJECT_ROOT / "tests" / "engine"
SRC_DIR = PROJECT_ROOT / "src" / "engine"
COMMON_DIR = SRC_DIR / "common"
UNITY_DIR = TESTS_DIR / "unity"

# Test configurations with their source files
TEST_CONFIGS = {
    "test_activations": {
        "sources": ["src/engine/common/activations.c"],
    },
    "test_attention": {
        "sources": [
            "src/engine/common/attention.c",
            "src/engine/common/activations.c",
        ],
    },
    "test_adversarial": {
        "sources": [
            "src/engine/fastenhancer.c",
            "src/engine/common/fft.c",
            "src/engine/common/stft.c",
            "src/engine/common/conv.c",
            "src/engine/common/gru.c",
            "src/engine/common/attention.c",
            "src/engine/common/activations.c",
            "src/engine/common/compression.c",
        ],
    },
    "test_benchmark_stats": {
        "sources": [],  # Header-only or minimal
    },
    "test_compression": {
        "sources": ["src/engine/common/compression.c"],
    },
    "test_conv": {
        "sources": ["src/engine/common/conv.c"],
    },
    "test_edge_cases": {
        "sources": [
            "src/engine/fastenhancer.c",
            "src/engine/common/fft.c",
            "src/engine/common/stft.c",
            "src/engine/common/conv.c",
            "src/engine/common/gru.c",
            "src/engine/common/attention.c",
            "src/engine/common/activations.c",
            "src/engine/common/compression.c",
        ],
    },
    "test_exports": {
        "sources": [
            "src/engine/exports.c",
            "src/engine/fastenhancer.c",
            "src/engine/common/fft.c",
            "src/engine/common/stft.c",
            "src/engine/common/conv.c",
            "src/engine/common/gru.c",
            "src/engine/common/attention.c",
            "src/engine/common/activations.c",
            "src/engine/common/compression.c",
        ],
    },
    "test_fast_math": {
        "sources": [
            "src/engine/fastenhancer.c",
            "src/engine/common/fft.c",
            "src/engine/common/stft.c",
            "src/engine/common/conv.c",
            "src/engine/common/gru.c",
            "src/engine/common/attention.c",
            "src/engine/common/activations.c",
            "src/engine/common/compression.c",
        ],
    },
    "test_fft": {
        "sources": ["src/engine/common/fft.c"],
    },
    "test_golden_vector": {
        "sources": [
            "src/engine/exports.c",
            "src/engine/fastenhancer.c",
            "src/engine/common/fft.c",
            "src/engine/common/stft.c",
            "src/engine/common/conv.c",
            "src/engine/common/gru.c",
            "src/engine/common/attention.c",
            "src/engine/common/activations.c",
            "src/engine/common/compression.c",
        ],
    },
    "test_gru": {
        "sources": [
            "src/engine/common/gru.c",
            "src/engine/common/activations.c",
        ],
    },
    "test_inference": {
        "sources": [
            "src/engine/fastenhancer.c",
            "src/engine/common/fft.c",
            "src/engine/common/stft.c",
            "src/engine/common/conv.c",
            "src/engine/common/gru.c",
            "src/engine/common/attention.c",
            "src/engine/common/activations.c",
            "src/engine/common/compression.c",
        ],
    },
    "test_pipeline": {
        "sources": ["src/engine/pipeline.c"],
    },
    "test_safety": {
        "sources": [
            "src/engine/fastenhancer.c",
            "src/engine/common/gru.c",
            "src/engine/common/compression.c",
            "src/engine/common/attention.c",
            "src/engine/common/activations.c",
            "src/engine/common/fft.c",
            "src/engine/common/stft.c",
            "src/engine/common/conv.c",
        ],
    },
    "test_simd_accuracy": {
        "sources": [
            "src/engine/fastenhancer.c",
            "src/engine/common/activations.c",
            "src/engine/common/attention.c",
            "src/engine/common/fft.c",
            "src/engine/common/stft.c",
            "src/engine/common/conv.c",
            "src/engine/common/gru.c",
            "src/engine/common/compression.c",
        ],
    },
    "test_stft": {
        "sources": [
            "src/engine/common/stft.c",
            "src/engine/common/fft.c",
        ],
    },
    "test_weight_load": {
        "sources": ["src/engine/exports.c"],
    },
}

def run_test(test_name: str, verbose: bool = False) -> Tuple[bool, str]:
    """Compile and run a single test via WSL gcc
    
    Returns:
        (success: bool, output: str)
    """
    try:
        # Create a shell script for WSL
        wsl_project_root = "/mnt/c/Users/famil/Desktop/fastenhancer"
        test_file = f"{wsl_project_root}/tests/engine/{test_name}.c"
        output_exe = f"/tmp/test_{test_name}"
        
        config = TEST_CONFIGS.get(test_name, {})
        sources = config.get("sources", [])
        
        # Build source file list
        source_files = [
            f"{wsl_project_root}/tests/engine/unity/unity.c",
            test_file,
        ]
        
        for src in sources:
            source_files.append(f"{wsl_project_root}/{src}")
        
        # Build compile command
        compile_cmd = (
            f"gcc -O2 -std=c11 "
            f"-I {wsl_project_root}/tests/engine/unity "
            f"-I {wsl_project_root}/src/engine/common "
            f"-I {wsl_project_root}/src/engine "
            f"-I {wsl_project_root}/src/engine/configs "
            f"{' '.join(source_files)} "
            f"-lm -o {output_exe} && {output_exe}"
        )
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"Compiling: {test_name}")
            print(f"Running via WSL...")
            print(f"{'='*70}")
        
        # Run via WSL
        result = subprocess.run(
            ["wsl", compile_cmd],
            capture_output=True,
            text=True,
            timeout=60,
        )
        
        output = result.stdout + result.stderr
        
        # Check for errors
        if "error:" in output.lower():
            return False, output
        
        if result.returncode != 0 and "error:" not in output.lower():
            # Check if it's a test failure (return code != 0 but no compile error)
            if "FAIL" not in output and "PASS" not in output:
                return False, f"Compilation or execution failed with code {result.returncode}\n{output}"
        
        # Parse Unity test output
        if "FAIL" in output or "Failures" in output:
            return False, output
        
        return True, output
        
    except subprocess.TimeoutExpired:
        return False, f"Test {test_name} timed out (60s)"
    except Exception as e:
        return False, f"EXCEPTION: {str(e)}"

def main():
    """Run all tests"""
    verbose = "--verbose" in sys.argv
    specific_test = None
    
    if "--test-name" in sys.argv:
        idx = sys.argv.index("--test-name")
        if idx + 1 < len(sys.argv):
            specific_test = sys.argv[idx + 1]
    
    print(f"Using WSL gcc for compilation")
    print(f"Project root: {PROJECT_ROOT}")
    
    # Determine which tests to run
    tests_to_run = [specific_test] if specific_test else sorted(TEST_CONFIGS.keys())
    
    # Run priority tests first (the 3 that were previously failing)
    priority_tests = ["test_fft", "test_golden_vector", "test_activations"]
    priority_to_run = [t for t in priority_tests if t in tests_to_run]
    other_to_run = [t for t in tests_to_run if t not in priority_tests]
    
    all_tests = priority_to_run + other_to_run
    
    results = {}
    failed_tests = []
    
    print(f"\n{'='*70}")
    print(f"Running {len(all_tests)} C Native Tests (via WSL gcc)")
    print(f"{'='*70}")
    
    for i, test_name in enumerate(all_tests, 1):
        print(f"\n[{i}/{len(all_tests)}] Running {test_name}...", end=" ", flush=True)
        
        success, output = run_test(test_name, verbose=verbose)
        results[test_name] = {"success": success, "output": output}
        
        if success:
            # Count passed tests from output
            if "test(s)" in output and "passed" in output:
                # Extract count from Unity summary
                import re
                match = re.search(r"(\d+) test.*?passed", output)
                if match:
                    test_count = int(match.group(1))
                    print(f"✓ PASS ({test_count} tests)")
                else:
                    print(f"✓ PASS")
            else:
                print(f"✓ PASS")
        else:
            print(f"✗ FAIL")
            failed_tests.append(test_name)
        
        if verbose and not success:
            print(output[:500])
    
    # Summary
    print(f"\n{'='*70}")
    print("TEST SUMMARY")
    print(f"{'='*70}")
    
    passed = len([t for t in results if results[t]["success"]])
    total = len(results)
    
    print(f"\nPassed: {passed}/{total}")
    
    if failed_tests:
        print(f"\nFailed tests ({len(failed_tests)}):")
        for test in failed_tests:
            print(f"  - {test}")
            if not verbose:
                print(f"    Output preview: {results[test]['output'][:200]}")
        return 1
    else:
        print("\n✓ All tests passed!")
        return 0

if __name__ == "__main__":
    sys.exit(main())
