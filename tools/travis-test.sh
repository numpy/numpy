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
werrors="-Werror=vla -Werror=nonnull -Werror=pointer-arith"
werrors="$werrors -Werror=implicit-function-declaration"

# build with c99 by default

setup_base()
{
  # use default python flags but remove sign-compare
  sysflags="$($PYTHON -c "from distutils import sysconfig; \
    print (sysconfig.get_config_var('CFLAGS'))")"
  export CFLAGS="$sysflags $werrors -Wlogical-op -Wno-sign-compare"
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
    # The job run with USE_DEBUG=1 on travis needs this.
    export CFLAGS=$CFLAGS" -Wno-maybe-uninitialized"
    $PYTHON setup.py build build_src --verbose-cfg build_ext --inplace 2>&1 | tee log
  fi
  grep -v "_configtest" log \
    | grep -vE "ld returned 1|no files found matching" \
    | grep -vE "no previously-included files matching" \
    | grep -vE "manifest_maker: standard file '-c'" \
    | grep -E "warning\>" \
    | tee warnings
  if [ "$LAPACK" != "None" ]; then
    [[ $(wc -l < warnings) -lt 1 ]]
  fi
}

run_test()
{
  # Install the test dependencies.
  # Clear PYTHONOPTIMIZE when running `pip install -r test_requirements.txt`
  # because version 2.19 of pycparser (a dependency of one of the packages
  # in test_requirements.txt) does not provide a wheel, and the source tar
  # file does not install correctly when Python's optimization level is set
  # to strip docstrings (see https://github.com/eliben/pycparser/issues/291).
  PYTHONOPTIMIZE="" $PIP install -r test_requirements.txt

  if [ -n "$USE_DEBUG" ]; then
    export PYTHONPATH=$PWD
  fi

  # pytest aborts when running --durations with python3.6-dbg, so only enable
  # it for non-debug tests. That is a cPython bug fixed in later versions of
  # python3.7 but python3.7-dbg is not currently available on travisCI.
  if [ -z "$USE_DEBUG" ]; then
    DURATIONS_FLAG="--durations 10"
  fi

  if [ -n "$RUN_COVERAGE" ]; then
    COVERAGE_FLAG=--coverage
  fi

  # We change directories to make sure that python won't find the copy
  # of numpy in the source directory.
  mkdir -p empty
  cd empty
  INSTALLDIR=$($PYTHON -c \
    "import os; import numpy; print(os.path.dirname(numpy.__file__))")
  export PYTHONWARNINGS=default

  if [ -n "$CHECK_BLAS" ]; then
    $PYTHON ../tools/openblas_support.py --check_version $OpenBLAS_version
  fi

  if [ -n "$RUN_FULL_TESTS" ]; then
    export PYTHONWARNINGS="ignore::DeprecationWarning:virtualenv"
    $PYTHON -b ../runtests.py -n -v --mode=full $DURATIONS_FLAG $COVERAGE_FLAG
  else
    $PYTHON ../runtests.py -n -v $DURATIONS_FLAG
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
    $PYTHON `which asv` check --python=same
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

if [ -n "$USE_WHEEL" ] && [ $# -eq 0 ]; then
  # ensure some warnings are not issued
  export CFLAGS=$CFLAGS" -Wno-sign-compare -Wno-unused-result"
  # adjust gcc flags if C coverage requested
  if [ -n "$RUN_COVERAGE" ]; then
     export NPY_DISTUTILS_APPEND_FLAGS=1
     export CC='gcc --coverage'
     export F77='gfortran --coverage'
     export F90='gfortran --coverage'
     export LDFLAGS='--coverage'
  fi
  $PYTHON setup.py build --warn-error build_src --verbose-cfg bdist_wheel
  # Make another virtualenv to install into
  virtualenv --python=`which $PYTHON` venv-for-wheel
  . venv-for-wheel/bin/activate
  # Move out of source directory to avoid finding local numpy
  pushd dist
  $PIP install --pre --no-index --upgrade --find-links=. numpy
  popd

  run_test

elif [ -n "$USE_SDIST" ] && [ $# -eq 0 ]; then
  # temporary workaround for sdist failures.
  $PYTHON -c "import fcntl; fcntl.fcntl(1, fcntl.F_SETFL, 0)"
  # ensure some warnings are not issued
  export CFLAGS=$CFLAGS" -Wno-sign-compare -Wno-unused-result"
  $PYTHON setup.py sdist
  # Make another virtualenv to install into
  virtualenv --python=`which $PYTHON` venv-for-wheel
  . venv-for-wheel/bin/activate
  # Move out of source directory to avoid finding local numpy
  pushd dist
  $PIP install numpy*
  popd
  run_test
else
  setup_base
  run_test
fi
