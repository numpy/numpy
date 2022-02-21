# This script is used by .github/workflows/wheels.yml to build wheels with
# cibuildwheel. It runs the full test suite, checks for lincense inclusion
# and that the openblas version is correct.
set -xe

PROJECT_DIR="$1"

python -c "import numpy; numpy.show_config()"
if [[ $RUNNER_OS == "Windows" ]]; then
    # GH 20391
    PY_DIR=$(python -c "import sys; print(sys.prefix)")
    mkdir $PY_DIR/libs
fi
python -c "import sys; import numpy; sys.exit(not numpy.test('full', extra_argv=['-vvv']))"

python $PROJECT_DIR/tools/wheels/check_license.py
if [[ $UNAME == "Linux" || $UNAME == "Darwin" ]] ; then
    python $PROJECT_DIR/tools/openblas_support.py --check_version
fi
