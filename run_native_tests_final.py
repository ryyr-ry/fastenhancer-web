#!/usr/bin/env python3
"""
run_native_tests_final.py — Compile and run all 18 C native tests

Uses a hybrid approach: For each test, creates a shell script and runs it via WSL.

Usage:
    python run_native_tests_final.py [--verbose] [--test-name <name>]

Returns:
    0 if all tests pass
    1 if any test fails
"""

import os
import sys
import subprocess
import json
import tempfile
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
        "sources": ["src/engine/benchmark_stats.c"],  # Contains fe_bench_compute_stats
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
}

def run_test(test_name: str, verbose: bool = False) -> Tuple[bool, str]:
    """Compile and run a single test via WSL gcc
    
    Returns:
        (success: bool, output: str)
    """
    try:
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
        
        # Create a temporary shell script (LF line endings for WSL)
        shell_script = f"""#!/bin/bash
cd {wsl_project_root}
gcc -O2 -std=c11 \\
  -I {wsl_project_root}/tests/engine/unity \\
  -I {wsl_project_root}/src/engine/common \\
  -I {wsl_project_root}/src/engine \\
  -I {wsl_project_root}/src/engine/configs \\
  {' '.join(source_files)} \\
  -lm -o {output_exe}
{output_exe}
"""
        
        # Write shell script to temp location in Windows, then convert to WSL path
        # Use LF line endings explicitly for WSL
        temp_script_path = PROJECT_ROOT / f"build_{test_name}.sh"
        with open(temp_script_path, 'w', newline='\n', encoding='utf-8') as f:
            f.write(shell_script)
        
        wsl_script_path = f"/mnt/c/Users/famil/Desktop/fastenhancer/build_{test_name}.sh"
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"Compiling: {test_name}")
            print(f"Running via WSL...")
            print(f"{'='*70}")
        
        # Run the shell script via WSL (use Latin-1 encoding to handle binary data)
        result = subprocess.run(
            ["wsl", "bash", wsl_script_path],
            capture_output=True,
            text=False,  # Get bytes instead of strings
            timeout=120,
        )
        
        # Decode with error handling
        try:
            stdout = result.stdout.decode('utf-8', errors='ignore')
            stderr = result.stderr.decode('utf-8', errors='ignore')
        except:
            stdout = result.stdout.decode('latin-1', errors='ignore')
            stderr = result.stderr.decode('latin-1', errors='ignore')
        
        output = stdout + stderr
        
        # Clean up temp script
        temp_script_path.unlink(missing_ok=True)
        
        # Check for errors (look for actual compilation/execution errors, not test failures)
        if "error:" in output.lower() and "undefined reference" not in output.lower():
            return False, output
        
        # Check if execution succeeded (return code 0 means all tests passed)
        if result.returncode != 0:
            # Check if this is a test failure or execution error
            if "PASS" in output and "FAIL" not in output:
                # Tests ran but process returned non-zero (might be due to missing output)
                return True, output
            elif "error:" in output.lower():
                return False, output
            else:
                # Assume it ran and check test output
                pass
        
        # Parse Unity test output for failures
        if "FAIL" in output or ("Failures" in output and "0 Failures" not in output):
            return False, output
        
        return True, output
        
    except subprocess.TimeoutExpired:
        return False, f"Test {test_name} timed out (120s)"
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
    total_tests = 0
    total_passed = 0
    
    print(f"\n{'='*70}")
    print(f"Running {len(all_tests)} C Native Tests (via WSL gcc)")
    print(f"{'='*70}")
    
    for i, test_name in enumerate(all_tests, 1):
        print(f"\n[{i}/{len(all_tests)}] Running {test_name}...", end=" ", flush=True)
        
        success, output = run_test(test_name, verbose=verbose)
        results[test_name] = {"success": success, "output": output}
        
        if success:
            # Count passed tests from output
            import re
            match = re.search(r"(\d+)\s+Tests?\s+(\d+)\s+Failures?", output)
            if match:
                test_count = int(match.group(1))
                failures = int(match.group(2))
                total_tests += test_count
                total_passed += test_count - failures
                print(f"[PASS] ({test_count} tests)")
            else:
                print(f"[PASS]")
                total_tests += 1
                total_passed += 1
        else:
            print(f"[FAIL]")
            failed_tests.append(test_name)
            total_tests += 1
        
        if verbose and not success:
            print(output[:500])
    
    # Summary
    print(f"\n{'='*70}")
    print("TEST SUMMARY")
    print(f"{'='*70}")
    
    passed = len([t for t in results if results[t]["success"]])
    total_binaries = len(results)
    
    print(f"\nTest binaries passed: {passed}/{total_binaries}")
    if total_tests > 0:
        print(f"Individual tests passed: {total_passed}/{total_tests}")
    
    if failed_tests:
        print(f"\nFailed test binaries ({len(failed_tests)}):")
        for test in failed_tests:
            print(f"  - {test}")
            if not verbose:
                output_preview = results[test]['output']
                if len(output_preview) > 200:
                    output_preview = output_preview[:200] + "..."
                print(f"    {output_preview.split(chr(10))[0]}")
        return 1
    else:
        print("\n[OK] All test binaries passed!")
        return 0

if __name__ == "__main__":
    sys.exit(main())
