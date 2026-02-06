#!/usr/bin/env python3
"""
Simple NumPy C-Level Tracing Example

Shows the complete call tree for a simple ufunc operation.

Build: spin build -- -Denable-ctrace=true
Run:   spin python ctrace_simple.py
"""

import numpy as np
from numpy._core._ctrace import CTrace, resolve_symbol, is_available

if not is_available():
    print("Error: Build with -Denable-ctrace=true")
    exit(1)

# Prepare arrays outside tracing
a = np.array([2**i - 1 for i in range(16)])

# Trace np.add
print("Tracing: np.bitwise_count(a)")
print("=" * 60)

def trace_callback(func, caller, depth, is_entry):
    name = resolve_symbol(func) or f"0x{func:x}"
    # Filter out noisy low-level helpers
    skip = ["Py_TYPE", "Py_INCREF", "Py_DECREF", "Py_XINCREF", "Py_XDECREF",
            "_Py_IsImmortal", "_Py_NewRef", "_ZL7Py_TYPE", "_ZL17PyType_Has",
            "Py_SIZE", "PyType_HasFeature", "Py_IS_TYPE"]
    if any(s in name for s in skip):
       return
    indent = "  " * min(depth, 20)
    arrow = ">" if is_entry else "<"
    print(f"{indent}{arrow} {name}")

with CTrace(callback=trace_callback):
    result = np.bitwise_count(a)

print("=" * 60)
print(f"Result: {result}")
