Regenerating lapack_lite source
===============================

:Authors: * David M. Cooke <cookedm@physics.mcmaster.ca>
          * Eric Wieser (upgraded lapack version on 2017-03-26)

The ``numpy/linalg/f2c_*.c`` files are ``f2c``'d versions of the LAPACK routines
required by the ``LinearAlgebra`` module, and wrapped by the ``lapack_lite``
module. The scripts in this directory can be used to create these files
automatically from a directory of LAPACK source files.

You'll need `plex 2.0.0dev`_, available from PyPI, installed to do the
appropriate scrubbing. As of writing, **this is only available for python 2.7**,
and is unlikely to ever be ported to python 3.

.. _plex 2.0.0dev: https://pypi.python.org/pypi/plex/

The routines that ``lapack_litemodule.c`` wraps are listed in
``wrapped_routines``, along with a few exceptions that aren't picked up
properly. Assuming that you have an unpacked LAPACK source tree in
``~/LAPACK``, you generate the new routines in a directory ``new-lite/`` with::

$ python2 ./make_lite.py wrapped_routines ~/LAPACK new-lite/

This will grab the right routines, with dependencies, put them into the
appropriate ``f2c_*.f`` files, run ``f2c`` over them, then do some scrubbing
similar to that done to generate the CLAPACK_ distribution.

.. _CLAPACK: http://netlib.org/clapack/index.html

The output C files in git use the LAPACK source from the LAPACK_ page, using
version 3.2.2. Unfortunately, newer versions use newer FORTRAN features, which
are increasingly not supported by ``f2c``. As these are found, the patch files
will need to be changed to re-express new constructs with legacy constructs.

.. _LAPACK: http://netlib.org/lapack/index.html
