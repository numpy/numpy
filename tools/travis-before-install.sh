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
  target=$(python tools/openblas_support.py)
  sudo cp -r $target/64/lib/* /usr/lib
  sudo cp $target/64/include/* /usr/include
fi

pip install --upgrade pip setuptools
pip install nose pytz cython pytest
if [ -n "$USE_ASV" ]; then pip install asv; fi
popd
