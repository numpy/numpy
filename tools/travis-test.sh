#!/bin/bash

set -ex

# Travis legacy boxes give you 1.5 CPUs, container-based boxes give you 2 CPUs
export NPY_NUM_BUILD_JOBS=2

# setup env
if [ -r /usr/lib/libeatmydata/libeatmydata.so ]; then
  # much faster package installation
  export LD_PRELOAD='/usr/lib/libeatmydata/libeatmydata.so'
elif [ -r /usr/lib/*/libeatmydata.so ]; then
  # much faster package installation
  export LD_PRELOAD='/usr/$LIB/libeatmydata.so'
fi

source builds/venv/bin/activate

# travis venv tests override python
PYTHON=${PYTHON:-python}
PIP=${PIP:-pip}

if [ -n "$PYTHON_OO" ]; then
  PYTHON="${PYTHON} -OO"
fi


if [ -n "$PY3_COMPATIBILITY_CHECK" ]; then
  PYTHON="${PYTHON} -3"
fi

# make some warnings fatal, mostly to match windows compilers
werrors="-Werror=declaration-after-statement -Werror=vla "
werrors+="-Werror=nonnull -Werror=pointer-arith"

setup_base()
{
  # We used to use 'setup.py install' here, but that has the terrible
  # behaviour that if a copy of the package is already installed in the
  # install location, then the new copy just gets dropped on top of it.
  # Travis typically has a stable numpy release pre-installed, and if we
  # don't remove it, then we can accidentally end up e.g. running old
  # test modules that were in the stable release but have been removed
  # from master. (See gh-2765, gh-2768.)  Using 'pip install' also has
  # the advantage that it tests that numpy is 'pip install' compatible,
  # see e.g. gh-2766...
  if [ -z "$USE_DEBUG" ]; then
    if [ -z "$IN_CHROOT" ]; then
      $PIP install .
    else
      sysflags="$($PYTHON -c "from distutils import sysconfig; \
        print (sysconfig.get_config_var('CFLAGS'))")"
      CFLAGS="$sysflags $werrors -Wlogical-op" $PIP install -v . 2>&1 | tee log
      grep -v "_configtest" log \
        | grep -vE "ld returned 1|no previously-included files matching|manifest_maker: standard file '-c'" \
        | grep -E "warning\>" \
        | tee warnings
      # Check for an acceptable number of warnings. Some warnings are out of
      # our control, so adjust the number as needed. At the moment a
      # cython generated code produces a warning about '-2147483648L', but
      # the code seems to compile OK.
      [[ $(wc -l < warnings) -lt 2 ]]
    fi
  else
    sysflags="$($PYTHON -c "from distutils import sysconfig; \
      print (sysconfig.get_config_var('CFLAGS'))")"
    CFLAGS="$sysflags $werrors" $PYTHON setup.py build_ext --inplace
  fi
}

setup_chroot()
{
  # this can all be replaced with:
  # apt-get install libpython2.7-dev:i386
  # CC="gcc -m32" LDSHARED="gcc -m32 -shared" LDFLAGS="-m32 -shared" \
  #   linux32 python setup.py build
  # when travis updates to ubuntu 14.04
  #
  # NumPy may not distinguish between 64 and 32 bit ATLAS in the
  # configuration stage.
  DIR=$1
  set -u
  sudo debootstrap --variant=buildd --include=fakeroot,build-essential \
    --arch=$ARCH --foreign $DIST $DIR
  sudo chroot $DIR ./debootstrap/debootstrap --second-stage

  # put the numpy repo in the chroot directory
  sudo rsync -a $TRAVIS_BUILD_DIR $DIR/

  # set up repos in the chroot directory for installing packages
  echo deb http://archive.ubuntu.com/ubuntu/ \
    $DIST main restricted universe multiverse \
    | sudo tee -a $DIR/etc/apt/sources.list
  echo deb http://archive.ubuntu.com/ubuntu/ \
    $DIST-updates main restricted universe multiverse \
    | sudo tee -a $DIR/etc/apt/sources.list
  echo deb http://security.ubuntu.com/ubuntu \
    $DIST-security  main restricted universe multiverse \
    | sudo tee -a $DIR/etc/apt/sources.list

  sudo chroot $DIR bash -c "apt-get update"
  # faster operation with preloaded eatmydata
  sudo chroot $DIR bash -c "apt-get install -qq -y --force-yes eatmydata"
  echo '/usr/$LIB/libeatmydata.so' | \
    sudo tee -a $DIR/etc/ld.so.preload

  # install needed packages
  sudo chroot $DIR bash -c "apt-get install -qq -y --force-yes \
    libatlas-dev libatlas-base-dev gfortran \
    python-dev python-nose python-pip cython"
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
  INSTALLDIR=$($PYTHON -c \
    "import os; import numpy; print(os.path.dirname(numpy.__file__))")
  export PYTHONWARNINGS=default
  if [ -n "$RUN_FULL_TESTS" ]; then
    $PYTHON ../tools/test-installed-numpy.py --mode=full
  else
    $PYTHON ../tools/test-installed-numpy.py
  fi
  if [ -n "$USE_ASV" ]; then
    pushd ../benchmarks
    $PYTHON `which asv` machine --machine travis
    $PYTHON `which asv` dev 2>&1| tee asv-output.log
    if grep -q Traceback asv-output.log; then
      echo "Some benchmarks have errors!"
      exit 1
    fi
    popd
  fi
}

export PYTHON
export PIP
$PIP install setuptools
if [ -n "$USE_WHEEL" ] && [ $# -eq 0 ]; then
  # Build wheel
  $PIP install wheel
  # ensure that the pip / setuptools versions deployed inside
  # the venv are recent enough
  $PIP install -U virtualenv
  $PYTHON setup.py bdist_wheel
  # Make another virtualenv to install into
  virtualenv --python=`which $PYTHON` venv-for-wheel
  . venv-for-wheel/bin/activate
  # Move out of source directory to avoid finding local numpy
  pushd dist
  pip install --pre --no-index --upgrade --find-links=. numpy
  pip install nose
  popd
  run_test
elif [ -n "$USE_SDIST" ] && [ $# -eq 0 ]; then
  # use an up-to-date pip / setuptools inside the venv
  $PIP install -U virtualenv
  $PYTHON setup.py sdist
  # Make another virtualenv to install into
  virtualenv --python=`which $PYTHON` venv-for-wheel
  . venv-for-wheel/bin/activate
  # Move out of source directory to avoid finding local numpy
  pushd dist
  pip install numpy*
  pip install nose
  popd
  run_test
elif [ -n "$USE_CHROOT" ] && [ $# -eq 0 ]; then
  DIR=/chroot
  setup_chroot $DIR
  # run again in chroot with this time testing
  sudo linux32 chroot $DIR bash -c \
    "cd numpy && PYTHON=python PIP=pip IN_CHROOT=1 $0 test"
else
  setup_base
  run_test
fi

