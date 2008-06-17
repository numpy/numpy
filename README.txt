NumPy is a replacement of Numeric Python that adds the features of numarray.
To install system-wide:

sudo python setup.py install

The setup.py script will take advantage of fast BLAS on your system if it can
find it.  You can guide the process using a site.cfg file.

If fast BLAS and LAPACK cannot be found, then a slower default version is used.

After installation, tests can be run (from outside the source
directory) with:

python -c 'import numpy; numpy.test()'

Please note that you must have the 'nose' test framework installed in order to
run the tests.  More information about nose is available here: 

http://somethingaboutorange.com/mrl/projects/nose/ 

The most current development version of NumPy is always available from our
subversion repository:

http://svn.scipy.org/svn/numpy/trunk
