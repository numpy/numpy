#! /bin/sh
# Script to build tarballs, windows and OS X installers on OS X

# Note that we build the corresponding set of OS X binaries to the python.org
# downloads, i.e. two versions for Python 2.7. The Intel 32/64-bit version is
# for OS X 10.6+, the other dmg installers are for 10.3+ and are built on 10.5

#---------------
# Build tarballs
#---------------
paver sdist


#--------------------
# Build documentation
#--------------------
# Check we're using the correct g++/c++ for the 32-bit 2.6 version we build for
# the docs and the 64-bit 2.7 dmg installer.
# We do this because for Python 2.6 we use a symlink on the PATH to select
# /usr/bin/g++-4.0, while for Python 2.7 we need the default 4.2 version.
export PATH=~/Code/tmp/gpp40temp/:$PATH
gpp="$(g++ --version | grep "4.0")"
if [ -z "$gpp" ]; then
    echo "Wrong g++ version, we need 4.0 to compile scipy with Python 2.6"
    exit 1
fi

# bootstrap needed to ensure we build the docs from the right scipy version
paver bootstrap
source bootstrap/bin/activate

# build pdf docs
paver pdf


#--------------------------------------------------------
# Build Windows and 64-bit OS X installers (on OS X 10.6)
#--------------------------------------------------------
export MACOSX_DEPLOYMENT_TARGET=10.6
# Use GCC 4.2 for 64-bit OS X installer for Python 2.7
export PATH=~/Code/tmp/gpp42temp/:$PATH
gpp="$(g++ --version | grep "4.2")"
if [ -z "$gpp" ]; then
    echo "Wrong g++ version, we need 4.2 for 64-bit binary for Python 2.7"
    exit 1
fi

paver dmg -p 2.7   # 32/64-bit version

paver bdist_superpack -p 3.2
paver bdist_superpack -p 3.1
paver bdist_superpack -p 2.7
paver bdist_superpack -p 2.6
paver bdist_superpack -p 2.5


#--------------------------------------------
# Build 32-bit OS X installers (on OS X 10.5)
#--------------------------------------------
#export MACOSX_DEPLOYMENT_TARGET=10.3
#paver dmg -p 2.6
#paver dmg -p 2.7  # 32-bit version
#export CC=/usr/bin/gcc-4.0  # necessary on 10.6, not sure about 10.5
#paver dmg -p 2.5


paver write_release_and_log


#-------------------------------------------------------
# Build basic (no SSE) Windows installers to put on PyPi
#-------------------------------------------------------
paver bdist_wininst_simple -p 2.5
paver bdist_wininst_simple -p 2.6
paver bdist_wininst_simple -p 2.7
paver bdist_wininst_simple -p 3.1
paver bdist_wininst_simple -p 3.2

