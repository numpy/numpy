#!/bin/bash

uname -a
free -m
df -h
ulimit -a
mkdir builds
pushd builds

# Build into own virtualenv
# We therefore control our own environment, avoid travis' numpy
#
# Some change in virtualenv 14.0.5 caused `test_f2py` to fail. So, we have
# pinned `virtualenv` to the last known working version to avoid this failure.
# Appears we had some issues with certificates on Travis. It looks like
# bumping to 14.0.6 will help.
pip install -U 'virtualenv==14.0.6'

if [ -n "$USE_DEBUG" ]
then
  virtualenv --python=python3-dbg venv
else
  virtualenv --python=python venv
fi

source venv/bin/activate
python -V
pip install --upgrade pip setuptools
pip install nose pytz cython
if [ -n "$USE_ASV" ]; then pip install asv; fi
popd
