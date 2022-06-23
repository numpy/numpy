.. _f2py-skbuild:

============================
Using via ``scikit-build``
============================

``scikit-build`` provides two separate concepts geared towards the users of Python extension modules.

1. A ``setuptools`` replacement (legacy behaviour)
2. A series of ``cmake`` modules with definitions which help building Python extensions

.. note::

   It is possible to use ``scikit-build``'s ``cmake`` modules to `bypass the
   cmake setup mechanism`_ completely, and to write targets which call ``f2py
   -c``. This usage is **not recommended** since the point of these build system
   documents are to move away from the internal ``numpy.distutils`` methods.

For situations where no ``setuptools`` replacements are required or wanted (i.e.
if ``wheels`` are not needed), it is recommended to instead use the vanilla
``cmake`` setup described in :ref:`f2py-cmake`.

Fibonacci Walkthrough (F77)
===========================

We will consider the ``fib``  example from :ref:`f2py-getting-started` section.

.. literalinclude:: ./../code/fib1.f
    :language: fortran

``CMake`` modules only
~~~~~~~~~~~~~~~~~~~~~~

Consider using the following ``CMakeLists.txt``.

.. literalinclude:: ./../code/CMakeLists_skbuild.txt
   :language: cmake

Much of the logic is the same as in :ref:`f2py-cmake`, however notably here the
appropriate module suffix is generated via ``sysconfig.get_config_var("SO")``.
The resulting extension can be built and loaded in the standard workflow.

.. code:: bash

    ls .
    # CMakeLists.txt fib1.f
    cmake -S . -B build
    cmake --build build
    cd build
    python -c "import numpy as np; import fibby; a = np.zeros(9); fibby.fib(a); print (a)"
    # [ 0.  1.  1.  2.  3.  5.  8. 13. 21.]


``setuptools`` replacement
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. note::

   **As of November 2021**

   The behavior described here of driving the ``cmake`` build of a module is
   considered to be legacy behaviour and should not be depended on.

The utility of ``scikit-build`` lies in being able to drive the generation of
more than extension modules, in particular a common usage pattern is the
generation of Python distributables (for example for PyPI).

The workflow with ``scikit-build`` straightforwardly supports such packaging requirements. Consider augmenting the project with a ``setup.py`` as defined:

.. literalinclude:: ./../code/setup_skbuild.py
   :language: python

Along with a commensurate ``pyproject.toml``

.. literalinclude:: ./../code/pyproj_skbuild.toml
   :language: toml

Together these can build the extension using ``cmake`` in tandem with other
standard ``setuptools`` outputs. Running ``cmake`` through ``setup.py`` is
mostly used when it is necessary to integrate with extension modules not built
with ``cmake``.

.. code:: bash

    ls .
    # CMakeLists.txt fib1.f pyproject.toml setup.py
    python setup.py build_ext --inplace
    python -c "import numpy as np; import fibby.fibby; a = np.zeros(9); fibby.fibby.fib(a); print (a)"
    # [ 0.  1.  1.  2.  3.  5.  8. 13. 21.]

Where we have modified the path to the module as ``--inplace`` places the
extension module in a subfolder.

.. _bypass the cmake setup mechanism: https://scikit-build.readthedocs.io/en/latest/cmake-modules/F2PY.html
