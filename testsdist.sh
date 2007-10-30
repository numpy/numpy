PREFIX=$PWD
SVNVER=4346
rm -rf $PREFIX/build
rm -rf $PREFIX/dist
python setup.py sdist
(cd $PREFIX/dist && tar -xzf numpy-1.0.4.dev$SVNVER.tar.gz)
(cd $PREFIX/dist/numpy-1.0.4.dev$SVNVER && python setup.py scons)

