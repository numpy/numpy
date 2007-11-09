PREFIX=$PWD
rm -rf $PREFIX/build
rm -rf $PREFIX/tmp
DEBUG_SCONS_CHECK=0 python setup.py scons --jobs=1 install --prefix=$PREFIX/tmp
(cd $PREFIX/tmp && PYTHONPATH=$PREFIX/tmp/lib/python2.4/site-packages python -c "import numpy; print numpy; numpy.test(level = 9999); numpy.show_config()")
