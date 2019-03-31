#!/usr/bin/env bash

apt-get -yq update
apt-get -yq install libatlas-dev libatlas-base-dev liblapack-dev
wget http://buildbot.pypy.org/nightly/py3.6/pypy-c-jit-latest-linux64.tar.bz2 -O pypy.tar.bz2
mkdir -p pypy3
(cd pypy3; tar --strip-components=1 -xf ../pypy.tar.bz2)
pypy3/bin/pypy3 -mensurepip
pypy3/bin/pypy3 -m pip install --upgrade pip setuptools
pypy3/bin/pypy3 -m pip install --user cython==0.29.0 pytest pytz --no-warn-script-location
pypy3/bin/pypy3 runtests.py -- -rsx --junitxml=junit/test-results.xml --durations 10
