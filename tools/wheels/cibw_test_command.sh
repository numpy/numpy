# This script is used by .github/workflows/wheels.yml to build wheels with
# cibuildwheel. It runs the full test suite, checks for lincense inclusion
# and that the openblas version is correct.
set -xe

PROJECT_DIR="$1"

python -m pip install threadpoolctl
python -c "import numpy; numpy.show_config()"
if [[ $RUNNER_OS == "Windows" ]]; then
    # GH 20391
    PY_DIR=$(python -c "import sys; print(sys.prefix)")
    mkdir $PY_DIR/libs
fi
if [[ $RUNNER_OS == "macOS" ]]; then
    # Install same version of gfortran as the openblas-libs builds
    if [[ $PLATFORM == "macosx-arm64" ]]; then
        PLAT="arm64"
    fi
    source $PROJECT_DIR/tools/wheels/gfortran_utils.sh
    install_gfortran
fi
# Set available memory value to avoid OOM problems on aarch64.
# See gh-22418.
export NPY_AVAILABLE_MEM="4 GB"
if [[ $(python -c "import sys; print(sys.implementation.name)") == "pypy" ]]; then
  # make PyPy more verbose, try to catc a segfault in
  # numpy/lib/tests/test_function_base.py
  python -c "import sys; import numpy; sys.exit(not numpy.test(label='full', verbose=2))"
else
  python -c "import sys; import numpy; sys.exit(not numpy.test(label='full'))"
fi
python $PROJECT_DIR/tools/wheels/check_license.py
python $PROJECT_DIR/tools/openblas_support.py --check_version
