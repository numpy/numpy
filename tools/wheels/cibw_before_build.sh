set -xe

PROJECT_DIR="$1"
UNAME="$(uname)"

# Update license
if [[ $UNAME == "Linux" ]] ; then
    cat $PROJECT_DIR/tools/wheels/LICENSE_linux.txt >> $PROJECT_DIR/LICENSE.txt
elif [[ $UNAME == "Darwin" ]]; then
    cat $PROJECT_DIR/tools/wheels/LICENSE_osx.txt >> $PROJECT_DIR/LICENSE.txt
fi

# Install Openblas
if [[ $UNAME == "Linux" || $UNAME == "Darwin" ]] ; then
    basedir=$(python tools/openblas_support.py)
    cp -r $basedir/lib/* /usr/local/lib
    cp $basedir/include/* /usr/local/include
fi

# Install GFortran
if [[ $UNAME == "Darwin" ]]; then
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
    # Manually symlink gfortran-4.9 to plain gfortran for f2py.
    # No longer needed after Feb 13 2020 as gfortran is already present
    # and the attempted link errors. Keep this for future reference.
    # ln -s /usr/local/bin/gfortran-4.9 /usr/local/bin/gfortran
fi
