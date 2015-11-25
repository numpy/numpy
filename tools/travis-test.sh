#!/bin/bash
set -ex

# Travis legacy boxes give you 1.5 CPUs, container-based boxes give you 2 CPUs
export NPY_NUM_BUILD_JOBS=2

# setup env
if [ -r /usr/lib/libeatmydata/libeatmydata.so ]; then
  # much faster package installation
  export LD_PRELOAD=/usr/lib/libeatmydata/libeatmydata.so
fi

# make some warnings fatal, mostly to match windows compilers
werrors="-Werror=declaration-after-statement -Werror=vla -Werror=nonnull"

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
  if [ -z "$IN_CHROOT" ]; then
    $PIP install .
  else
    sysflags="$($PYTHON -c "from distutils import sysconfig; print (sysconfig.get_config_var('CFLAGS'))")"
    CFLAGS="$sysflags $werrors -Wlogical-op" $PIP install . 2>&1 | tee log
    grep -v "_configtest" log | grep -vE "ld returned 1|no previously-included files matching" | grep -E "warning\>";
    # accept a mysterious memset warning that shows with -flto
    test $(grep -v "_configtest" log | grep -vE "ld returned 1|no previously-included files matching" | grep -E "warning\>" -c) -lt 2;
  fi
else
  sysflags="$($PYTHON -c "from distutils import sysconfig; print (sysconfig.get_config_var('CFLAGS'))")"
  CFLAGS="$sysflags $werrors" $PYTHON setup.py build_ext --inplace
fi
}

setup_chroot()
{
  # this can all be replaced with:
  # apt-get install libpython2.7-dev:i386
  # CC="gcc -m32" LDSHARED="gcc -m32 -shared" LDFLAGS="-m32 -shared" linux32 python setup.py build
  # when travis updates to ubuntu 14.04
  DIR=$1
  set -u
  sudo debootstrap --variant=buildd --include=fakeroot,build-essential --arch=$ARCH --foreign $DIST $DIR
  sudo chroot $DIR ./debootstrap/debootstrap --second-stage
  sudo rsync -a $TRAVIS_BUILD_DIR $DIR/
  echo deb http://archive.ubuntu.com/ubuntu/ $DIST main restricted universe multiverse | sudo tee -a $DIR/etc/apt/sources.list
  echo deb http://archive.ubuntu.com/ubuntu/ $DIST-updates main restricted universe multiverse | sudo tee -a $DIR/etc/apt/sources.list
  echo deb http://security.ubuntu.com/ubuntu $DIST-security  main restricted universe multiverse | sudo tee -a $DIR/etc/apt/sources.list
  sudo chroot $DIR bash -c "apt-get update"
  sudo chroot $DIR bash -c "apt-get install -qq -y --force-yes eatmydata"
  echo /usr/lib/libeatmydata/libeatmydata.so | sudo tee -a $DIR/etc/ld.so.preload
  sudo chroot $DIR bash -c "apt-get install -qq -y --force-yes libatlas-dev libatlas-base-dev gfortran python3-dev python3-nose python3-pip cython3 cython"
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
  PYTHON=python3-dbg
fi

if [ -n "$PYTHON_OO" ]; then
  PYTHON="$PYTHON -OO"
fi

export PYTHON
export PIP
if [ -n "$USE_WHEEL" ] && [ $# -eq 0 ]; then
  # Build wheel
  $PIP install wheel
  $PYTHON setup.py bdist_wheel
  # Make another virtualenv to install into
  virtualenv --python=python venv-for-wheel
  . venv-for-wheel/bin/activate
  # Move out of source directory to avoid finding local numpy
  pushd dist
  $PIP install --pre --no-index --upgrade --find-links=. numpy
  $PIP install nose
  popd
  run_test
elif [ "$USE_CHROOT" != "1" ]; then
  setup_base
  run_test
elif [ -n "$USE_CHROOT" ] && [ $# -eq 0 ]; then
  DIR=/chroot
  setup_chroot $DIR
  # run again in chroot with this time testing
  sudo linux32 chroot $DIR bash -c "cd numpy && PYTHON=python3 PIP=pip3 IN_CHROOT=1 $0 test"
else
  run_test
fi

