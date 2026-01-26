# This script is used by .github/workflows/wheels.yml to run the full test
# suite, and that the openblas version is correct.
set -xe

PROJECT_DIR="$1"

python -c "import numpy; numpy.show_config()"

if [[ $RUNNER_OS == "Windows" && $IS_32_BIT == true ]] ; then
  # Avoid this in GHA: "ERROR: Found GNU link.exe instead of MSVC link.exe"
  rm /c/Program\ Files/Git/usr/bin/link.EXE
fi

# Set available memory value to avoid OOM problems on aarch64 (see gh-22418)
export NPY_AVAILABLE_MEM="4 GB"

FREE_THREADED_BUILD="$(python -c"import sysconfig; print(bool(sysconfig.get_config_var('Py_GIL_DISABLED')))")"
if [[ $FREE_THREADED_BUILD == "True" ]]; then
    # Manually check that importing NumPy does not re-enable the GIL.
    # In principle the tests should catch this but it seems harmless to leave it
    # here as a final sanity check before uploading broken wheels
    if [[ $(python -c "import numpy" 2>&1) == *"The global interpreter lock (GIL) has been enabled"* ]]; then
        echo "Error: Importing NumPy re-enables the GIL in the free-threaded build"
        exit 1
    fi
fi

# Run full tests with -n=auto. This makes pytest-xdist distribute tests across
# the available N CPU cores. Also print the durations for the 10 slowest tests
# to help with debugging slow or hanging tests
python -c "import sys; import numpy; sys.exit(not numpy.test(label='full', extra_argv=['-n=auto', '--durations=10']))"
