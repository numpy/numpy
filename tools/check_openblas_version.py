"""
usage: check_openblas_version.py <min_version>

Check the blas version is blas from scipy-openblas and is higher than
min_version
example: check_openblas_version.py 0.3.26
"""

import pprint
import sys

import numpy

version = sys.argv[1]
deps = numpy.show_config('dicts')['Build Dependencies']
assert "blas" in deps
print("Build Dependencies: blas")
pprint.pprint(deps["blas"])
assert deps["blas"]["version"].split(".") >= version.split(".")
assert deps["blas"]["name"] == "scipy-openblas"
