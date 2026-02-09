#!/usr/bin/env python3
"""
Simple NumPy C-Level Tracing Example

Shows the complete call tree for a simple ufunc operation,
including SIMD-optimized inner loops.

Build: spin build -- -Denable-ctrace=true
Run:   spin python ctrace_simple.py
"""

import numpy as np
from numpy._core._ctrace import CTrace, is_available

if not is_available():
    print("Error: Build with -Denable-ctrace=true")
    exit(1)

# Show CPU features
try:
    from numpy._core._multiarray_umath import __cpu_baseline__
    print(f"CPU baseline: {__cpu_baseline__}")
except ImportError:
    pass
print()

# Prepare arrays outside tracing - use float64 for SIMD paths
a = np.arange(5)

# Trace np.sqrt which has SIMD-optimized paths
print("Tracing: np.bitwise_count(a) - showing SIMD inner loop")
print("=" * 60)

with CTrace() as ct:
    result = np.bitwise_count(a)

# Print the trace using utility methods
ct.print_trace(show_exit=True)

print("=" * 60)
print()

# Print summary and hotspots
ct.print_summary()
print()
ct.print_hotspots(top_n=5)

print()
print(f"Result[:5]: {result[:5]}")
