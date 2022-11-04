#!/usr/bin/env python
#
# Example stub for running `python -m dev.py`
#
# Copy this into your project root.

import os
import sys
import runpy

sys.path.remove(os.path.abspath(os.path.dirname(sys.argv[0])))
try:
    runpy.run_module("dev.py", run_name="__main__")
except ImportError:
    print("Cannot import dev.py; please install it using")
    print()
    print(
        "  pip install git+https://github.com/scientific-python/dev.py@main#egg=dev.py"
    )
    print()
    sys.exit(1)
