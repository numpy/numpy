# This script is used by .github/workflows/wheels.yml to build wheels with
# cibuildwheel. It runs the full test suite, checks for lincense inclusion
# and that the openblas version is correct.
set -xe

PROJECT_DIR="$1"
UNAME="$(uname)"

python -c "import numpy; numpy.show_config()"
python -c "import sys; import numpy; sys.exit(not numpy.test('full', extra_argv=['-vv']))"

python $PROJECT_DIR/tools/wheels/check_license.py
if [[ $UNAME == "Linux" || $UNAME == "Darwin" ]] ; then
    python $PROJECT_DIR/tools/openblas_support.py --check_version
fi
