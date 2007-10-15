PREFIX=/usr/media/src/src/dsp/bzrversion/numpy.scons/tmp
rm -rf build
rm -rf $PREFIX
python setupscons.py scons
python setupscons.py build
#--prefix=$PREFIX
##python setup.py install --prefix=$PREFIX
#(cd $PREFIX && PYTHONPATH=$PREFIX/lib/python2.5/site-packages python -c "import numpy; print numpy; numpy.test(level = 9999)")
