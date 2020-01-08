#!/bin/bash

uname -a
free -m
df -h
ulimit -a

if [ -n "$DOWNLOAD_OPENBLAS" ]; then
  pwd
  ls -ltrh
  target=$(python tools/openblas_support.py)
  sudo cp -r $target/lib/* /usr/lib
  sudo cp $target/include/* /usr/include
fi

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

popd

pip install --upgrade pip setuptools
pip install -r test_requirements.txt
if [ -n "$USE_ASV" ]; then pip install asv; fi
