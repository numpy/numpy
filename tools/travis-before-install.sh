#!/bin/bash

# Exit the script immediately if a command exits with a non-zero status,
# and print commands and their arguments as they are executed.
set -ex

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
  virtualenv --python=$(which python3-dbg) venv
else
  virtualenv --python=python venv
fi

source venv/bin/activate
python -V
gcc --version

popd

pip install --upgrade pip

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
# urllib3 is needed for openblas_support
pip install setuptools wheel urllib3 `grep cython test_requirements.txt`

if [ -n "$DOWNLOAD_OPENBLAS" ]; then
  pwd
  target=$(python tools/openblas_support.py)
  sudo cp -r $target/lib/* /usr/lib
  sudo cp $target/include/* /usr/include
fi


if [ -n "$USE_ASV" ]; then pip install asv; fi
