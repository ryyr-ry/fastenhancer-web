#!/usr/bin/env python3
"""
run_native_tests.py — Compile and run all 18 C native tests using Clang

Clang is used from emscripten SDK for native x86_64 Windows compilation.
Tests use Unity framework (tests/engine/unity/) for C testing.

Usage:
    python run_native_tests.py [--verbose] [--test-name <name>]

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
CLANG = r"C:\Users\famil\emsdk\upstream\bin\clang.exe"
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
            "src/engine/pipeline.c",
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
        "sources": ["src/engine/benchmark_stats.c"],
    },
    "test_compression": {
        "sources": ["src/engine/common/compression.c"],
    },
    "test_conv": {
        "sources": ["src/engine/common/conv.c"],
    },
    "test_edge_cases": {
        "sources": [
            "src/engine/pipeline.c",
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
            "src/engine/pipeline.c",
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
            "src/engine/pipeline.c",
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
            "src/engine/pipeline.c",
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
            "src/engine/pipeline.c",
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
            "src/engine/pipeline.c",
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
            "src/engine/pipeline.c",
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

def build_compile_command(test_name: str) -> List[str]:
    """Build clang compile command for a test"""
    test_file = TESTS_DIR / f"{test_name}.c"
    output_exe = PROJECT_ROOT / f"{test_name}.exe"
    
    config = TEST_CONFIGS.get(test_name, {})
    sources = config.get("sources", [])
    
    # Build command
    cmd = [
        CLANG,
        "-O2",
        "-std=c11",
        "-D_CRT_SECURE_NO_WARNINGS",
        "-D_NORETURN=",  # Work around clang stdnoreturn.h conflict with Windows SDK
        f"-I{TESTS_DIR / 'unity'}",
        f"-I{COMMON_DIR}",
        f"-I{SRC_DIR}",
        f"-I{SRC_DIR / 'configs'}",
        str(UNITY_DIR / "unity.c"),
        str(test_file),
    ]
    
    # Add source files
    for src in sources:
        cmd.append(str(PROJECT_ROOT / src))
    
    # Link math library (Clang on Windows doesn't need -lm explicitly usually,
    # but we'll try without it first and add if needed)
    # cmd.append("-lm")
    
    cmd.append(f"-o{output_exe}")
    
    return cmd

def run_test(test_name: str, verbose: bool = False) -> Tuple[bool, str]:
    """Compile and run a single test
    
    Returns:
        (success: bool, output: str)
    """
    try:
        output_exe = PROJECT_ROOT / f"{test_name}.exe"
        
        # Build compile command
        cmd = build_compile_command(test_name)
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"Compiling: {test_name}")
            print(f"Command: {' '.join(cmd)}")
            print(f"{'='*70}")
        
        # Compile
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
        )
        
        if result.returncode != 0:
            error_msg = result.stderr or result.stdout
            return False, f"COMPILE ERROR:\n{error_msg}"
        
        # Run
        if verbose:
            print(f"Running: {output_exe}")
        
        result = subprocess.run(
            [str(output_exe)],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
        )
        
        output = result.stdout + result.stderr
        
        # Parse Unity test output
        if "FAIL" in output or result.returncode != 0:
            return False, output
        
        return True, output
        
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
    
    # Check Clang exists
    if not Path(CLANG).exists():
        print(f"ERROR: Clang not found at {CLANG}")
        return 1
    
    print(f"Using Clang: {CLANG}")
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
    print(f"Running {len(all_tests)} C Native Tests")
    print(f"{'='*70}")
    
    for i, test_name in enumerate(all_tests, 1):
        print(f"\n[{i}/{len(all_tests)}] Running {test_name}...", end=" ", flush=True)
        
        success, output = run_test(test_name, verbose=verbose)
        results[test_name] = {"success": success, "output": output}
        
        if success:
            # Count passed tests from output
            test_count = output.count(" OK\n") + output.count(" ok\n")
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
            print(f"✗ FAIL")
            failed_tests.append(test_name)
        
        if verbose and not success:
            print(output)
    
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
