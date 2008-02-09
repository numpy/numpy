# PREFIX=$PWD
# rm -rf $PREFIX/build
# rm -rf $PREFIX/tmp
# MKL=None python setupscons.py scons --jobs=4 install --prefix=$PREFIX/tmp
# (cd $PREFIX/tmp && PYTHONPATH=$PREFIX/tmp/lib/python2.5/site-packages python -c "import numpy; print numpy; numpy.test(level = 9999); numpy.show_config()")

PREFIX=$PWD
rm -rf $PREFIX/build
rm -rf $PREFIX/tmp
python setupscons.py scons --jobs=4 install --prefix=$PREFIX/tmp
(cd $PREFIX/tmp && PYTHONPATH=$PREFIX/tmp/lib/python2.5/site-packages python -c "import numpy; print numpy; numpy.test(level = 9999); numpy.show_config()")

# PREFIX=$PWD
# #rm -rf $PREFIX/build
# #rm -rf $PREFIX/tmp
# MKL=None python setupscons.py scons --jobs=4 --silent=2 install --prefix=$PREFIX/tmp
# (cd $PREFIX/tmp/lib/python2.5/site-packages/numpy/distutils/scons/tests/f2pyext/ && \
#  PYTHONPATH=$PREFIX/tmp/lib/python2.5/site-packages python setup.py scons)
