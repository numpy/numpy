"""
Check the blas version is blas and is higher than min_version
usage: check_openblas_version.py <min_version>
example: check_openblas_version.py 0.3.26
"""

import numpy
import pprint
import sys

version = sys.argv[1]
deps = numpy.show_config('dicts')['Build Dependencies']
assert "blas" in deps
print("Build Dependencies: blas")
pprint.pprint(deps["blas"])
print(f"Checking verions >= {version}")
assert deps["blas"]["version"].split(".") >= version.split(".")
