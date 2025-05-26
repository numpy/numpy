set -xe

PROJECT_DIR="${1:-$PWD}"


# remove any cruft from a previous run
rm -rf build

# Update license
echo "" >> $PROJECT_DIR/LICENSE.txt
echo "----" >> $PROJECT_DIR/LICENSE.txt
echo "" >> $PROJECT_DIR/LICENSE.txt
cat $PROJECT_DIR/LICENSES_bundled.txt >> $PROJECT_DIR/LICENSE.txt
if [[ $RUNNER_OS == "Linux" ]] ; then
    cat $PROJECT_DIR/tools/wheels/LICENSE_linux.txt >> $PROJECT_DIR/LICENSE.txt
elif [[ $RUNNER_OS == "macOS" ]]; then
    cat $PROJECT_DIR/tools/wheels/LICENSE_osx.txt >> $PROJECT_DIR/LICENSE.txt
elif [[ $RUNNER_OS == "Windows" ]]; then
    cat $PROJECT_DIR/tools/wheels/LICENSE_win32.txt >> $PROJECT_DIR/LICENSE.txt
fi

if [[ $(python -c"import sys; print(sys.maxsize)") < $(python -c"import sys; print(2**33)") ]]; then
    echo "No BLAS used for 32-bit wheels"
    export INSTALL_OPENBLAS=false
elif [ -z $INSTALL_OPENBLAS ]; then
    # the macos_arm64 build might not set this variable
    export INSTALL_OPENBLAS=true
fi

# Install Openblas from scipy-openblas64
if [[ "$INSTALL_OPENBLAS" = "true" ]] ; then
    # by default, use scipy-openblas64
    OPENBLAS=openblas64
    # Possible values for RUNNER_ARCH in github are
    # X86, X64, ARM, or ARM64
    # TODO: should we detect a missing RUNNER_ARCH and use platform.machine()
    #    when wheel build is run outside github?
    # On 32-bit platforms, use scipy_openblas32
    # On win-arm64 use scipy_openblas32
    if [[ $RUNNER_ARCH == "X86" || $RUNNER_ARCH == "ARM" ]] ; then
        OPENBLAS=openblas32
    elif [[ $RUNNER_ARCH == "ARM64" && $RUNNER_OS == "Windows" ]] ; then
        OPENBLAS=openblas32
    fi
    echo PKG_CONFIG_PATH is $PKG_CONFIG_PATH, OPENBLAS is ${OPENBLAS}
    PKG_CONFIG_PATH=$PROJECT_DIR/.openblas
    rm -rf $PKG_CONFIG_PATH
    mkdir -p $PKG_CONFIG_PATH
    python -m pip install -r requirements/ci_requirements.txt
    python -c "import scipy_${OPENBLAS}; print(scipy_${OPENBLAS}.get_pkg_config())" > $PKG_CONFIG_PATH/scipy-openblas.pc
    # Copy the shared objects to a path under $PKG_CONFIG_PATH, the build
    # will point $LD_LIBRARY_PATH there and then auditwheel/delocate-wheel will
    # pull these into the wheel. Use python to avoid windows/posix problems
    python <<EOF
import os, scipy_${OPENBLAS}, shutil
srcdir = os.path.join(os.path.dirname(scipy_${OPENBLAS}.__file__), "lib")
shutil.copytree(srcdir, os.path.join("$PKG_CONFIG_PATH", "lib"))
srcdir = os.path.join(os.path.dirname(scipy_${OPENBLAS}.__file__), ".dylibs")
if os.path.exists(srcdir):  # macosx delocate
    shutil.copytree(srcdir, os.path.join("$PKG_CONFIG_PATH", ".dylibs"))
EOF
    # pkg-config scipy-openblas --print-provides
fi
if [[ $RUNNER_OS == "Windows" ]]; then
    # delvewheel is the equivalent of delocate/auditwheel for windows.
    python -m pip install delvewheel wheel
fi
