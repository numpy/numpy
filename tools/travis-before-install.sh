#!/bin/bash

uname -a
free -m
df -h
ulimit -a

if [ -n "$DOWNLOAD_OPENBLAS" ]; then
  pwd
  ls -ltrh
  target=$(python tools/openblas_support.py)
  if [ -d "$target/usr/local" ]; then
      sudo cp -r $target/usr/local/lib/* /usr/lib
      sudo cp $target/usr/local/include/* /usr/include
  else
      sudo cp -r $target/64/lib/* /usr/lib
      sudo cp $target/64/include/* /usr/include
  fi
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
