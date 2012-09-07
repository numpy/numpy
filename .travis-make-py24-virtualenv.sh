#!/bin/sh

VIRTENV=$1

set -x
set -e

curl -O http://www.python.org/ftp/python/2.4.6/Python-2.4.6.tar.bz2
tar xjf Python-2.4.6.tar.bz2
cd Python-2.4.6
cat >setup.cfg <<EOF
[build_ext]
library_dirs=/usr/lib/$(dpkg-architecture -qDEB_HOST_MULTIARCH)/
EOF
./configure --prefix=$PWD/install
make
make install
# This is the last version of virtualenv to support python 2.4:
curl -O https://raw.github.com/pypa/virtualenv/1.7.2/virtualenv.py
# And this is the last version of pip to support python 2.4. If
# there's a file matching "^pip-.*(zip|tar.gz|tar.bz2|tgz|tbz)$" in
# the current directory then virtualenv will take that as the pip
# source distribution to install
curl -O http://pypi.python.org/packages/source/p/pip/pip-1.1.tar.gz
install/bin/python2.4 ./virtualenv.py --distribute $VIRTENV
