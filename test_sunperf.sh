INSTALL_PREFIX=/home/david/numpy.sunperf/tmp
SUNSTUDIOPATH=/opt/sun/sunstudio12
PATH=$SUNSTUDIOPATH/bin/:$PATH

LD_LIBRARY_PATH=$SUNSTUDIOPATH/lib/:$SUNSTUDIOPATH/rtlibs/:$LD_LIBRARY_PATH
SUNPERF=$SUNSTUDIOPATH

#ATLAS=None
#BLAS=None
#LAPACK=None

rm -rf $INSTALL_PREFIX
rm -rf build
ATLAS=$ATLAS BLAS=$BLAS LAPACK=$LAPACK SUNPERF=$SUNPERF python setup.py config --compiler=sun --fcompiler=sun
ATLAS=$ATLAS BLAS=$BLAS LAPACK=$LAPACK SUNPERF=$SUNPERF python setup.py build --compiler=sun --fcompiler=sun

python setup.py install --prefix=$INSTALL_PREFIX
echo "======================================"
echo "              TESTING "
(cd tmp && LD_LIBRARY_PATH=$LD_LIBRARY_PATH PYTHONPATH=$INSTALL_PREFIX/lib/python2.5/site-packages python -c "import numpy; numpy.test()")
