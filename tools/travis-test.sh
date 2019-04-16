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

if [ -n "$PYTHON_OPTS" ]; then
  PYTHON="${PYTHON} $PYTHON_OPTS"
fi

# make some warnings fatal, mostly to match windows compilers
werrors="-Werror=declaration-after-statement -Werror=vla "
werrors+="-Werror=nonnull -Werror=pointer-arith"

# build with c99 by default

setup_base()
{
  # use default python flags but remoge sign-compare
  sysflags="$($PYTHON -c "from distutils import sysconfig; \
    print (sysconfig.get_config_var('CFLAGS'))")"
  export CFLAGS="$sysflags $werrors -Wlogical-op -Wno-sign-compare"
  # use c99
  export CFLAGS=$CFLAGS" -std=c99"
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
    $PIP install -v . 2>&1 | tee log
  else
    # Python3.5-dbg on travis seems to need this
    export CFLAGS=$CFLAGS" -Wno-maybe-uninitialized"
    $PYTHON setup.py build_ext --inplace 2>&1 | tee log
  fi
  grep -v "_configtest" log \
    | grep -vE "ld returned 1|no previously-included files matching|manifest_maker: standard file '-c'" \
    | grep -E "warning\>" \
    | tee warnings
  if [ "$LAPACK" != "None" ]; then
    [[ $(wc -l < warnings) -lt 1 ]]
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
  sudo chroot $DIR bash -c "apt-get install -qq -y eatmydata"
  echo '/usr/$LIB/libeatmydata.so' | \
    sudo tee -a $DIR/etc/ld.so.preload

  # install needed packages
  sudo chroot $DIR bash -c "apt-get install -qq -y \
    libatlas-base-dev gfortran python3-dev python3-pip \
    cython  python3-pytest"
}

run_test()
{
  if [ -n "$USE_DEBUG" ]; then
    export PYTHONPATH=$PWD
  fi

  if [ -n "$RUN_COVERAGE" ]; then
    $PIP install pytest-cov
    export NUMPY_EXPERIMENTAL_ARRAY_FUNCTION=1
    COVERAGE_FLAG=--coverage
  fi

  # We change directories to make sure that python won't find the copy
  # of numpy in the source directory.
  mkdir -p empty
  cd empty
  INSTALLDIR=$($PYTHON -c \
    "import os; import numpy; print(os.path.dirname(numpy.__file__))")
  export PYTHONWARNINGS=default
  if [ -n "$RUN_FULL_TESTS" ]; then
    export PYTHONWARNINGS="ignore::DeprecationWarning:virtualenv"
    $PYTHON ../tools/test-installed-numpy.py -v --mode=full $COVERAGE_FLAG
  else
    $PYTHON ../tools/test-installed-numpy.py -v
  fi

  if [ -n "$RUN_COVERAGE" ]; then
    # move back up to the source dir because we want to execute
    # gcov on the source files after the tests have gone through
    # the code paths
    cd ..

    # execute gcov on source files
    find . -name '*.gcno' -type f -exec gcov -pb {} +

    # move the C line coverage report files to the same path
    # as the Python report data
    mv *.gcov empty

    # move back to the previous path for good measure
    # as the Python coverage data is there
    cd empty

    # Upload coverage files to codecov
    bash <(curl -s https://codecov.io/bash) -X gcov -X coveragepy
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
  # ensure some warnings are not issued
  export CFLAGS=$CFLAGS" -Wno-sign-compare -Wno-unused-result"
  # use c99
  export CFLAGS=$CFLAGS" -std=c99"
  # adjust gcc flags if C coverage requested
  if [ -n "$RUN_COVERAGE" ]; then
     export NPY_DISTUTILS_APPEND_FLAGS=1
     export CC='gcc --coverage'
     export F77='gfortran --coverage'
     export F90='gfortran --coverage'
     export LDFLAGS='--coverage'
  fi
  $PYTHON setup.py bdist_wheel
  # Make another virtualenv to install into
  virtualenv --python=`which $PYTHON` venv-for-wheel
  . venv-for-wheel/bin/activate
  # Move out of source directory to avoid finding local numpy
  pushd dist
  $PIP install --pre --no-index --upgrade --find-links=. numpy
  $PIP install nose pytest

  if [ -n "$INSTALL_PICKLE5" ]; then
    $PIP install pickle5
  fi

  popd
  run_test
elif [ -n "$USE_SDIST" ] && [ $# -eq 0 ]; then
  # use an up-to-date pip / setuptools inside the venv
  $PIP install -U virtualenv
  # temporary workaround for sdist failures.
  $PYTHON -c "import fcntl; fcntl.fcntl(1, fcntl.F_SETFL, 0)"
  # ensure some warnings are not issued
  export CFLAGS=$CFLAGS" -Wno-sign-compare -Wno-unused-result"
  # use c99
  export CFLAGS=$CFLAGS" -std=c99"
  $PYTHON setup.py sdist
  # Make another virtualenv to install into
  virtualenv --python=`which $PYTHON` venv-for-wheel
  . venv-for-wheel/bin/activate
  # Move out of source directory to avoid finding local numpy
  pushd dist
  $PIP install numpy*
  $PIP install nose pytest
  if [ -n "$INSTALL_PICKLE5" ]; then
    $PIP install pickle5
  fi

  popd
  run_test
elif [ -n "$USE_CHROOT" ] && [ $# -eq 0 ]; then
  DIR=/chroot
  setup_chroot $DIR
  # the chroot'ed environment will not have the current locale,
  # avoid any warnings which may disturb testing
  export LANG=C LC_ALL=C
  # run again in chroot with this time testing with python3
  sudo linux32 chroot $DIR bash -c \
    "cd numpy && PYTHON=python3 PIP=pip3 IN_CHROOT=1 $0 test"
else
  setup_base
  run_test
fi
