#!/bin/bash

uname -a
free -m
df -h
ulimit -a
mkdir builds
pushd builds

# Build into own virtualenv
# We therefore control our own environment, avoid travis' numpy
pip install -U virtualenv

if [ -n "$USE_DEBUG" ]
then
  virtualenv --python=python3-dbg venv
else
  virtualenv --python=python venv
fi

source venv/bin/activate
python -V

if [ -n "$INSTALL_PICKLE5" ]; then
  pip install pickle5
fi

if [ -n "$PPC64_LE" ]; then
  # build script for POWER8 OpenBLAS available here:
  # https://github.com/tylerjereddy/openblas-static-gcc/blob/master/power8
  # built on GCC compile farm machine named gcc112
  # manually uploaded tarball to an unshared Dropbox location
  wget -O openblas-power8.tar.gz https://www.dropbox.com/s/zcwhk7c2zptwy0s/openblas-v0.3.5-ppc64le-power8.tar.gz?dl=0
  tar zxvf openblas-power8.tar.gz
  sudo cp -r ./64/lib/* /usr/lib
  sudo cp ./64/include/* /usr/include
fi

pip install --upgrade pip setuptools
pip install nose pytz cython pytest
if [ -n "$USE_ASV" ]; then pip install asv; fi
popd
