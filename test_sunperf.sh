INSTALL_PREFIX=/usr/media/src/src/dsp/numpy/numpy.sunperf/tmp
SUNSTUDIOPATH=$HOME/opt/sun/sunstudio12
PATH=$SUNSTUDIOPATH/bin/:$PATH

#ATLAS=None
#BLAS=None
#LAPACK=None
SUNPERF=$SUNSTUDIOPATH
LD_LIBRARY_PATH=$SUNSTUDIOPATH/lib

rm -rf $INSTALL_PREFIX
rm -rf build
ATLAS=$ATLAS BLAS=$BLAS LAPACK=$LAPACK SUNPERF=$SUNPERF python setup.py config --compiler=sun --fcompiler=sun
ATLAS=$ATLAS BLAS=$BLAS LAPACK=$LAPACK SUNPERF=$SUNPERF python setup.py build --compiler=sun --fcompiler=sun

python setup.py install --prefix=$INSTALL_PREFIX
echo "======================================"
echo "              TESTING "
(cd tmp && PYTHONPATH=$INSTALL_PREFIX/lib/python2.5/site-packages python -c "import numpy; numpy.test()")
