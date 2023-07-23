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

pip install --upgrade pip 'setuptools<49.2.0' wheel

# 'setuptools', 'wheel' and 'cython' are build dependencies.  This information
# is stored in pyproject.toml, but there is not yet a standard way to install
# those dependencies with, say, a pip command, so we'll just hard-code their
# installation here.  We only need to install them separately for the cases
# where numpy is installed with setup.py, which is the case for the Travis jobs
# where the environment variables USE_DEBUG or USE_WHEEL are set. When pip is
# used to install numpy, pip gets the build dependencies from pyproject.toml.
# A specific version of cython is required, so we read the cython package
# requirement using `grep cython test_requirements.txt` instead of simply
# writing 'pip install setuptools wheel cython'.
pip install `grep cython test_requirements.txt`

if [ -n "$DOWNLOAD_OPENBLAS" ]; then
  pwd
  target=$(python tools/openblas_support.py)
  sudo cp -r $target/lib/* /usr/lib
  sudo cp $target/include/* /usr/include
fi


if [ -n "$USE_ASV" ]; then pip install asv; fi
