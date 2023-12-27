set -xe

PROJECT_DIR="${1:-$PWD}"

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
    PKG_CONFIG_PATH=$PROJECT_DIR/.openblas
    rm -rf $PKG_CONFIG_PATH
    mkdir -p $PKG_CONFIG_PATH
    python -m pip install scipy-openblas64
    python -c "import scipy_openblas64; print(scipy_openblas64.get_pkg_config())" > $PKG_CONFIG_PATH/scipy-openblas.pc
    # Copy the shared objects to a path under $PKG_CONFIG_PATH, the build
    # will point $LD_LIBRARY_PATH there and then auditwheel/delocate-wheel will
    # pull these into the wheel. Use python to avoid windows/posix problems
    python <<EOF
import os, scipy_openblas64, shutil
srcdir = os.path.join(os.path.dirname(scipy_openblas64.__file__), "lib")
shutil.copytree(srcdir, os.path.join("$PKG_CONFIG_PATH", "lib"))
srcdir = os.path.join(os.path.dirname(scipy_openblas64.__file__), ".dylibs")
if os.path.exists(srcdir):  # macosx delocate
    shutil.copytree(srcdir, os.path.join("$PKG_CONFIG_PATH", ".dylibs"))
EOF
    # pkg-config scipy-openblas --print-provides
fi
if [[ $RUNNER_OS == "Windows" ]]; then
    # delvewheel is the equivalent of delocate/auditwheel for windows.
    python -m pip install delvewheel
fi

if [ -z $INSTALL_GFORTRAN ]; then
    # the macos_arm64 build might set this variable
    export INSTALL_GFORTRAN=true
fi

if [[ $RUNNER_OS == "macOS" && ("$INSTALL_GFORTRAN" = true) ]]; then
    # Install same version of gfortran as the openblas-libs builds
    PLAT=$(uname -m)

    source $PROJECT_DIR/tools/wheels/gfortran_utils.sh
    install_gfortran
    pip install "delocate>=0.10.4"
fi
