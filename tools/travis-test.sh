#!/bin/sh
set -ex

# setup env
if [ -r /usr/lib/libeatmydata/libeatmydata.so ]; then
  # much faster package installation
  export LD_PRELOAD=/usr/lib/libeatmydata/libeatmydata.so
fi


setup_base()
{
  # We used to use 'setup.py install' here, but that has the terrible
  # behaviour that if a copy of the package is already installed in
  # the install location, then the new copy just gets dropped on top
  # of it. Travis typically has a stable numpy release pre-installed,
  # and if we don't remove it, then we can accidentally end up
  # e.g. running old test modules that were in the stable release but
  # have been removed from master. (See gh-2765, gh-2768.)  Using 'pip
  # install' also has the advantage that it tests that numpy is 'pip
  # install' compatible, see e.g. gh-2766...
if [ -z "$USE_DEBUG" ]; then
  $PIP install .
else
  $PYTHON setup.py build_ext --inplace
fi
}

setup_chroot()
{
  # this can all be replaced with:
  # apt-get install libpython2.7-dev:i386
  # CC="gcc -m32" LDSHARED="gcc -m32 -shared" LDFLAGS="-m32 -shared" linux32 python setup.py build
  # when travis updates to ubuntu 14.04
  DIR=$1
  # speeds up setup as we don't have eatmydata during bootstrap
  sudo mkdir -p $DIR
  sudo mount -t tmpfs -o size=4G tmpfs $DIR
  set -u
  sudo apt-get -qq -y --force-yes install debootstrap eatmydata
  sudo debootstrap --variant=buildd --include=fakeroot,build-essential --arch=$ARCH --foreign $DIST $DIR
  sudo chroot $DIR ./debootstrap/debootstrap --second-stage
  sudo rsync -a $TRAVIS_BUILD_DIR $DIR/
  echo deb http://archive.ubuntu.com/ubuntu/ $DIST main restricted universe multiverse | sudo tee -a $DIR/etc/apt/sources.list
  echo deb http://archive.ubuntu.com/ubuntu/ $DIST-updates main restricted universe multiverse | sudo tee -a $DIR/etc/apt/sources.list
  echo deb http://security.ubuntu.com/ubuntu $DIST-security  main restricted universe multiverse | sudo tee -a $DIR/etc/apt/sources.list
  sudo chroot $DIR bash -c "apt-get update"
  sudo chroot $DIR bash -c "apt-get install -qq -y --force-yes eatmydata"
  echo /usr/lib/libeatmydata/libeatmydata.so | sudo tee -a $DIR/etc/ld.so.preload
  sudo chroot $DIR bash -c "apt-get install -qq -y --force-yes libatlas-dev libatlas-base-dev gfortran python3-dev python3-nose python3-pip"
}

setup_bento()
{
  export CI_ROOT=$PWD
  cd ..

  # Waf
  wget http://waf.googlecode.com/files/waf-1.7.13.tar.bz2
  tar xjvf waf-1.7.13.tar.bz2
  cd waf-1.7.13
  python waf-light
  export WAFDIR=$PWD
  cd ..

  # Bento
  wget https://github.com/cournape/Bento/archive/master.zip
  unzip master.zip
  cd Bento-master
  python bootstrap.py
  export BENTO_ROOT=$PWD
  cd ..

  cd $CI_ROOT

  # In-place numpy build
  $BENTO_ROOT/bentomaker build -v -i -j

  # Prepend to PYTHONPATH so tests can be run
  export PYTHONPATH=$PWD:$PYTHONPATH
}

run_test()
{
  if [ -n "$USE_DEBUG" ]; then
    export PYTHONPATH=$PWD
  fi
  # We change directories to make sure that python won't find the copy
  # of numpy in the source directory.
  mkdir -p empty
  cd empty
  INSTALLDIR=$($PYTHON -c "import os; import numpy; print(os.path.dirname(numpy.__file__))")
  export PYTHONWARNINGS=default
  $PYTHON ../tools/test-installed-numpy.py # --mode=full
  # - coverage run --source=$INSTALLDIR --rcfile=../.coveragerc $(which $PYTHON) ../tools/test-installed-numpy.py
  # - coverage report --rcfile=../.coveragerc --show-missing
}

# travis venv tests override python
PYTHON=${PYTHON:-python}
PIP=${PIP:-pip}

if [ -n "$USE_DEBUG" ]; then
  sudo apt-get install -qq -y --force-yes python3-dbg python3-dev python3-nose
  PYTHON=python3-dbg
fi

export PYTHON
export PIP
if [ "$USE_CHROOT" != "1" ] && [ "$USE_BENTO" != "1" ]; then
  setup_base
  run_test
elif [ -n "$USE_CHROOT" ] && [ $# -eq 0 ]; then
  DIR=/chroot
  setup_chroot $DIR
  # run again in chroot with this time testing
  sudo linux32 chroot $DIR bash -c "cd numpy && PYTHON=python3 PIP=pip3 $0 test"
elif [ -n "$USE_BENTO" ] && [ $# -eq 0 ]; then
  setup_bento
  # run again this time testing
  $0 test
else
  run_test
fi

