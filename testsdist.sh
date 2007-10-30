PREFIX=$PWD
SVNVER=` svn info | grep Revision | tr -d \ | cut -d: -f 2`
rm -rf $PREFIX/build
rm -rf $PREFIX/dist
python setup.py sdist
(cd $PREFIX/dist && tar -xzf numpy-1.0.4.dev$SVNVER.tar.gz)
(cd $PREFIX/dist/numpy-1.0.4.dev$SVNVER && python setup.py scons --jobs=3 install --prefix=$PREFIX/dist/tmp)
(cd $PREFIX/dist/tmp && PYTHONPATH=$PREFIX/dist/tmp/lib/python2.5/site-packages python -c "import numpy; numpy.test(level = 9999)")

