PREFIX=$PWD
rm -rf $PREFIX/build
rm -rf $PREFIX/tmp
python setup.py scons --jobs=4 install --prefix=$PREFIX/tmp
(cd $PREFIX/tmp && PYTHONPATH=$PREFIX/tmp/lib/python2.5/site-packages python -c "import numpy; print numpy; numpy.test(level = 9999); numpy.show_config()")
