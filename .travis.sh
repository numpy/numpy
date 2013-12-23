#!/bin/sh
# this can all be replaced with::
# apt-get install libpython2.7-dev:i386
# CC="gcc -m32" LDSHARED="gcc -m32 -shared" LDFLAGS="-m32 -shared" linux32 python setup.py build
# when travis updates to ubuntu 14.04

export LD_PRELOAD=/usr/lib/libeatmydata/libeatmydata.so
if [ $# -eq 0 ]; then
  sudo apt-get -qq debootstrap eatmydata
  DIR=/chroot
  sudo debootstrap --variant=buildd --include=fakeroot,build-essential --arch=i386 --foreign saucy $DIR
  sudo chroot $DIR ./debootstrap/debootstrap --second-stage
  sudo rsync -a $TRAVIS_BUILD_DIR $DIR/
  echo deb http://de.archive.ubuntu.com/ubuntu/ saucy main restricted universe multiverse | sudo tee -a $DIR/etc/apt/sources.list
  echo deb http://de.archive.ubuntu.com/ubuntu/ saucy-updates main restricted universe multiverse | sudo tee -a $DIR/etc/apt/sources.list
  echo deb http://security.ubuntu.com/ubuntu saucy-security  main restricted universe multiverse | sudo tee -a $DIR/etc/apt/sources.list
  sudo chroot $DIR bash -c "apt-get update"
  sudo chroot $DIR bash -c "apt-get install -qq -y --force-yes eatmydata"
  sudo chroot $DIR bash -c "cd numpy && ./.travis.sh test"
else
  apt-get install -qq -y --force-yes python-dev python-nose
  python setup.py build_ext --inplace
  python -c "import numpy; numpy.test()"
fi

