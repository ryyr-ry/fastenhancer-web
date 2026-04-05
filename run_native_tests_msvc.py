#!/usr/bin/env python3
"""
run_native_tests_msvc.py — Compile and run all 18 C native tests using MSVC

MSVC (cl.exe) is used from Visual Studio Build Tools for Windows native compilation.
Tests use Unity framework (tests/engine/unity/) for C testing.

Usage:
    python run_native_tests_msvc.py [--verbose] [--test-name <name>]

Returns:
    0 if all tests pass
    1 if any test fails
"""

import os
import sys
import subprocess
import json
import glob as globmod
from pathlib import Path
from typing import List, Dict, Tuple, Optional

PROJECT_ROOT = Path(__file__).parent


def _find_msvc_tools() -> Tuple[str, str, str, str]:
    """Auto-detect MSVC cl.exe, link.exe, lib dir, and Windows SDK ucrt lib dir."""
    vs_base = Path(r"C:\Program Files (x86)\Microsoft Visual Studio")
    if not vs_base.exists():
        vs_base = Path(r"C:\Program Files\Microsoft Visual Studio")

    cl_path: Optional[str] = None
    for edition in ["BuildTools", "Community", "Professional", "Enterprise"]:
        for year in ["2022", "2019"]:
            msvc_root = vs_base / year / edition / "VC" / "Tools" / "MSVC"
            if msvc_root.exists():
                versions = sorted(msvc_root.iterdir(), reverse=True)
                for ver_dir in versions:
                    candidate = ver_dir / "bin" / "Hostx64" / "x64" / "cl.exe"
                    if candidate.exists():
                        cl_path = str(candidate)
                        link_path = str(candidate.parent / "link.exe")
                        lib_path = str(ver_dir / "lib" / "x64")
                        break
                if cl_path:
                    break
        if cl_path:
            break

    if not cl_path:
        raise FileNotFoundError("MSVC cl.exe not found. Install Visual Studio Build Tools.")

    sdk_base = Path(r"C:\Program Files (x86)\Windows Kits\10\Lib")
    sdk_lib: Optional[str] = None
    if sdk_base.exists():
        sdk_versions = sorted(sdk_base.iterdir(), reverse=True)
        for sdk_ver in sdk_versions:
            ucrt = sdk_ver / "ucrt" / "x64"
            if ucrt.exists():
                sdk_lib = str(ucrt)
                break

    if not sdk_lib:
        raise FileNotFoundError("Windows SDK ucrt libs not found.")

    return cl_path, link_path, lib_path, sdk_lib


MSVC_CL, MSVC_LINK, MSVC_LIBS, WIN_SDK_LIBS = _find_msvc_tools()

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

def build_compile_command(test_name: str) -> List[str]:
    """Build MSVC compile command for a test"""
    test_file = TESTS_DIR / f"{test_name}.c"
    output_obj_dir = PROJECT_ROOT / "obj_temp"
    output_obj_dir.mkdir(exist_ok=True)
    
    config = TEST_CONFIGS.get(test_name, {})
    sources = config.get("sources", [])
    
    # Build compile command - compile to object files
    compile_sources = [
        str(UNITY_DIR / "unity.c"),
        str(test_file),
    ]
    
    # Add source files
    for src in sources:
        compile_sources.append(str(PROJECT_ROOT / src))
    
    cmd = [
        MSVC_CL,
        "/O2",
        "/std:c11",
        "/D_CRT_SECURE_NO_WARNINGS",
        f"/I{TESTS_DIR / 'unity'}",
        f"/I{COMMON_DIR}",
        f"/I{SRC_DIR}",
        f"/I{SRC_DIR / 'configs'}",
        "/Fo{0}{1}\\".format(output_obj_dir, "\\"),  # Object output directory
        "/c",  # Compile only
    ] + compile_sources
    
    return cmd

def build_link_command(test_name: str, obj_dir: Path) -> List[str]:
    """Build MSVC link command for a test"""
    output_exe = PROJECT_ROOT / f"{test_name}.exe"
    
    # Collect object files
    obj_files = list(obj_dir.glob("*.obj"))
    
    cmd = [
        MSVC_LINK,
        "/OUT:" + str(output_exe),
        "/SUBSYSTEM:CONSOLE",
        f"/LIBPATH:{MSVC_LIBS}",
        f"/LIBPATH:{WIN_SDK_LIBS}",
        "libcmt.lib",  # C runtime library
        "kernel32.lib",
    ] + [str(obj) for obj in obj_files]
    
    return cmd

def run_test(test_name: str, verbose: bool = False) -> Tuple[bool, str]:
    """Compile and run a single test
    
    Returns:
        (success: bool, output: str)
    """
    try:
        output_exe = PROJECT_ROOT / f"{test_name}.exe"
        obj_dir = PROJECT_ROOT / "obj_temp"
        obj_dir.mkdir(exist_ok=True)
        
        # Clean old objects
        for obj in obj_dir.glob("*.obj"):
            obj.unlink()
        
        # Build compile command
        compile_cmd = build_compile_command(test_name)
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"Compiling: {test_name}")
            print(f"Command: {' '.join(compile_cmd)}")
            print(f"{'='*70}")
        
        # Compile
        result = subprocess.run(
            compile_cmd,
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
        )
        
        if result.returncode != 0:
            error_msg = result.stderr or result.stdout
            return False, f"COMPILE ERROR:\n{error_msg}"
        
        # Link
        link_cmd = build_link_command(test_name, obj_dir)
        
        if verbose:
            print(f"Linking...")
        
        result = subprocess.run(
            link_cmd,
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
        )
        
        if result.returncode != 0:
            error_msg = result.stderr or result.stdout
            return False, f"LINK ERROR:\n{error_msg}"
        
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
    
    # Check MSVC exists
    if not Path(MSVC_CL).exists():
        print(f"ERROR: MSVC cl.exe not found at {MSVC_CL}")
        return 1
    
    print(f"Using MSVC cl.exe: {MSVC_CL}")
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
    print(f"Running {len(all_tests)} C Native Tests (MSVC)")
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
