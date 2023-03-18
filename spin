#!/usr/bin/env python
#
# Example stub for running `python -m spin`
#
# Copy this into your project root.

import os
import sys
import runpy

sys.path.remove(os.path.abspath(os.path.dirname(sys.argv[0])))
try:
    runpy.run_module("spin", run_name="__main__")
except ImportError:
    print("Cannot import spin; please install it using")
    print()
    print("  pip install spin")
    print()
    sys.exit(1)
