#!/bin/bash

# Exit the script immediately if a command exits with a non-zero status,
# and print commands and their arguments as they are executed.
set -ex

uname -a
free -m
df -h
ulimit -a

sudo apt update
sudo apt install gfortran eatmydata libgfortran5

if [ "$USE_DEBUG" ]
then
    sudo apt install python3-dbg python3-dev python3-setuptools
fi

mkdir builds
pushd builds

# Build into own virtualenv
# We therefore control our own environment, avoid travis' numpy

if [ -n "$USE_DEBUG" ]
then
  python3-dbg -m venv venv
else
  python -m venv venv
fi

source venv/bin/activate
python -V
gcc --version

popd

if [ -n "$DOWNLOAD_OPENBLAS" ]; then
  target=$(python tools/openblas_support.py)
  sudo cp -r $target/lib/* /usr/lib
  sudo cp $target/include/* /usr/include
fi



