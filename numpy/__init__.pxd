# NumPy static imports for Cython < 3.0
#
# DO NOT USE OR REFER TO THIS HEADER
#
# This is provided only to generate an error message on older Cython
# versions.
#
# See __init__.cython-30.pxd for the real Cython header
#

# intentionally created compiler error that only triggers on Cython < 3.0.0
DEF err = int('Build aborted: the NumPy Cython headers require Cython 3.0.0 or newer.')
