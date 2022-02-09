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
    mkdir -p openblas
    cp $target openblas
fi

# Install GFortran
if [[ $RUNNER_OS == "macOS" ]]; then
    # same version of gfortran as the openblas-libs and numpy-wheel builds
    curl -L https://github.com/MacPython/gfortran-install/raw/master/archives/gfortran-4.9.0-Mavericks.dmg -o gfortran.dmg
    GFORTRAN_SHA256=$(shasum -a 256 gfortran.dmg)
    KNOWN_SHA256="d2d5ca5ba8332d63bbe23a07201c4a0a5d7e09ee56f0298a96775f928c3c4b30  gfortran.dmg"
    if [ "$GFORTRAN_SHA256" != "$KNOWN_SHA256" ]; then
        echo sha256 mismatch
        exit 1
    fi

    hdiutil attach -mountpoint /Volumes/gfortran gfortran.dmg
    sudo installer -pkg /Volumes/gfortran/gfortran.pkg -target /
    otool -L /usr/local/gfortran/lib/libgfortran.3.dylib

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
