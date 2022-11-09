set -xe

PROJECT_DIR="$1"
PLATFORM=$(PYTHONPATH=tools python -c "import openblas_support; print(openblas_support.get_plat())")

# Update license
if [[ $RUNNER_OS == "Linux" ]] ; then
    cat $PROJECT_DIR/tools/wheels/LICENSE_linux.txt >> $PROJECT_DIR/LICENSE.txt
elif [[ $RUNNER_OS == "macOS" ]]; then
    cat $PROJECT_DIR/tools/wheels/LICENSE_osx.txt >> $PROJECT_DIR/LICENSE.txt
elif [[ $RUNNER_OS == "Windows" ]]; then
    cat $PROJECT_DIR/tools/wheels/LICENSE_win32.txt >> $PROJECT_DIR/LICENSE.txt
fi

# Install Openblas
if [[ $RUNNER_OS == "Linux" || $RUNNER_OS == "macOS" ]] ; then
    basedir=$(python tools/openblas_support.py)
    cp -r $basedir/lib/* /usr/local/lib
    cp $basedir/include/* /usr/local/include
    if [[ $RUNNER_OS == "macOS" && $PLATFORM == "macosx-arm64" ]]; then
        sudo mkdir -p /opt/arm64-builds/lib /opt/arm64-builds/include
        sudo chown -R $USER /opt/arm64-builds
        cp -r $basedir/lib/* /opt/arm64-builds/lib
        cp $basedir/include/* /opt/arm64-builds/include
    fi
elif [[ $RUNNER_OS == "Windows" ]]; then
    PYTHONPATH=tools python -c "import openblas_support; openblas_support.make_init('numpy')"
    target=$(python tools/openblas_support.py)
    ls /tmp
    mkdir -p openblas
    # bash on windows does not like cp -r $target/* openblas
    for f in $(ls $target); do
        cp -r $target/$f openblas
    done
    ls openblas
fi

# Install GFortran
if [[ $RUNNER_OS == "macOS" ]]; then
    # same version of gfortran as the openblas-libs and numpy-wheel builds
    local arch="x86_64"
    local type="native"
    curl -L https://github.com/isuruf/gcc/releases/download/gcc-11.3.0-2/gfortran-darwin-${arch}-${type}.tar.gz -o gfortran.dmg
    GFORTRAN_SHA=$(shasum  gfortran.dmg)
    KNOWN_SHA="c469a420d2d003112749dcdcbe3c684eef42127e  gfortran.dmg"
    if [ "$GFORTRAN_SHA256" != "$KNOWN_SHA256" ]; then
        echo sha256 mismatch
        exit 1
    fi

    hdiutil attach -mountpoint /Volumes/gfortran gfortran.dmg
    sudo installer -pkg /Volumes/gfortran/gfortran.pkg -target /
    otool -L /usr/local/gfortran/lib/libgfortran.5.dylib

    # arm64 stuff from gfortran_utils
    if [[ $PLATFORM == "macosx-arm64" ]]; then
        source $PROJECT_DIR/tools/wheels/gfortran_utils.sh
        install_arm64_cross_gfortran
    fi

    # Manually symlink gfortran-4.9 to plain gfortran for f2py.
    # No longer needed after Feb 13 2020 as gfortran is already present
    # and the attempted link errors. Keep this for future reference.
    # ln -s /usr/local/bin/gfortran-4.9 /usr/local/bin/gfortran
fi
