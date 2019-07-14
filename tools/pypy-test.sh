#!/usr/bin/env bash

# Exit if a command fails
set -e
set -o pipefail
# Print expanded commands
set -x

sudo apt-get -yq update
sudo apt-get -yq install libatlas-base-dev liblapack-dev gfortran-5
F77=gfortran-5 F90=gfortran-5 \

# Download the proper OpenBLAS x64 precompiled library
target=$(python tools/openblas_support.py)
echo getting OpenBLAS into $target
export LD_LIBRARY_PATH=$target/usr/local/lib
export LIB=$target/usr/local/lib
export INCLUDE=$target/usr/local/include

# Use a site.cfg to build with local openblas
cat << EOF > site.cfg
[openblas]
libraries = openblas
library_dirs = $target/usr/local/lib:$LIB
include_dirs = $target/usr/local/lib:$LIB
runtime_library_dirs = $target/usr/local/lib
EOF

echo getting PyPy 3.6 nightly
wget -q http://buildbot.pypy.org/nightly/py3.6/pypy-c-jit-latest-linux64.tar.bz2 -O pypy.tar.bz2
mkdir -p pypy3
(cd pypy3; tar --strip-components=1 -xf ../pypy.tar.bz2)
pypy3/bin/pypy3 -mensurepip
pypy3/bin/pypy3 -m pip install --upgrade pip setuptools
pypy3/bin/pypy3 -m pip install --user cython==0.29.0 pytest pytz --no-warn-script-location

echo
echo pypy3 version 
pypy3/bin/pypy3 -c "import sys; print(sys.version)"
echo

pypy3/bin/pypy3 runtests.py --show-build-log -- -rsx \
      --junitxml=junit/test-results.xml --durations 10

echo Make sure the correct openblas has been linked in

pypy3/bin/pip install .
pypy3/bin/pypy3 tools/openblas_support.py --check_version "$OpenBLAS_version"
