.. _building-blas-and-lapack:

BLAS and LAPACK
===============

.. _blas-lapack-selection:

Default behavior for BLAS and LAPACK selection
----------------------------------------------

When a NumPy build is invoked, BLAS and LAPACK library detection happens
automatically. The build system will attempt to locate a suitable library,
and try a number of known libraries in a certain order - most to least
performant. A typical order is: MKL, Accelerate, OpenBLAS, FlexiBLAS, BLIS,
plain ``libblas``/``liblapack``. This may vary per platform or over releases.
That order, and which libraries are tried, can be changed through the
``blas-order`` and ``lapack-order`` build options, for example::

    $ python -m pip install . -C-Dblas-order=openblas,mkl,blis -C-Dlapack-order=openblas,mkl,lapack

The first suitable library that is found will be used. In case no suitable
library is found, the NumPy build will print a warning and then use (slow!)
NumPy-internal fallback routines. In order to disallow use of those slow routines,
the ``allow-noblas`` build option can be used::

    $ python -m pip install . -C-Dallow-noblas=false

By default the LP64 (32-bit integer) interface to BLAS and LAPACK will be used.
For building against the ILP64 (64-bit integer) interface, one must use the
``use-ilp64`` build option::

    $ python -m pip install . -C-Duse-ilp64=true


.. _accelerated-blas-lapack-libraries:

Selecting specific BLAS and LAPACK libraries
--------------------------------------------

The ``blas`` and ``lapack`` build options are set to "auto" by default, which
means trying all known libraries. If you want to use a specific library, you
can set these build options to the library name (typically the lower-case name
that ``pkg-config`` expects). For example, to select plain ``libblas`` and
``liblapack`` (this is typically Netlib BLAS/LAPACK on Linux distros, and can
be dynamically switched between implementations on conda-forge), use::

    $ # for a development build
    $ spin build -C-Dblas=blas -C-Dlapack=lapack

    $ # to build and install a wheel
    $ python -m build -Csetup-args=-Dblas=blas -Csetup-args=-Dlapack=lapack
    $ pip install dist/numpy*.whl

    $ # Or, with pip>=23.1, this works too:
    $ python -m pip install . -Csetup-args=-Dblas=blas -Csetup-args=-Dlapack=lapack

Other options that should work (as long as they're installed with
``pkg-config`` support; otherwise they may still be detected but things are
inherently more fragile) include ``openblas``, ``mkl``, ``accelerate``,
``atlas`` and ``blis``.


Using pkg-config to detect libraries in a nonstandard location
--------------------------------------------------------------

The way BLAS and LAPACK detection works under the hood is that Meson tries
to discover the specified libraries first with ``pkg-config``, and then
with CMake. If all you have is a standalone shared library file (e.g.,
``armpl_lp64.so`` in ``/a/random/path/lib/`` and a corresponding header
file in ``/a/random/path/include/``), then what you have to do is craft
your own pkg-config file. It should have a matching name (so in this
example, ``armpl_lp64.pc``) and may be located anywhere. The
``PKG_CONFIG_PATH`` environment variable should be set to point to the
location of the ``.pc`` file. The contents of that file should be::

    libdir=/path/to/library-dir      # e.g., /a/random/path/lib
    includedir=/path/to/include-dir  # e.g., /a/random/path/include
    version=1.2.3                    # set to actual version
    extralib=-lm -lpthread -lgfortran   # if needed, the flags to link in dependencies
    Name: armpl_lp64
    Description: ArmPL - Arm Performance Libraries
    Version: ${version}
    Libs: -L${libdir} -larmpl_lp64      # linker flags
    Libs.private: ${extralib}
    Cflags: -I${includedir}

To check that this works as expected, you should be able to run::

    $ pkg-config --libs armpl_lp64
    -L/path/to/library-dir -larmpl_lp64
    $ pkg-config --cflags armpl_lp64
    -I/path/to/include-dir


Full list of BLAS and LAPACK related build options
--------------------------------------------------

BLAS and LAPACK are complex dependencies. Some libraries have more options that
are exposed via build options (see ``meson_options.txt`` in the root of the
repo for all of NumPy's build options).

- ``blas``: name of the BLAS library to use (default: ``auto``),
- ``lapack``: name of the LAPACK library to use (default: ``auto``),
- ``allow-noblas``: whether or not to allow building without external
  BLAS/LAPACK libraries (default: ``true``),
- ``blas-order``: order of BLAS libraries to try detecting (default may vary per platform),
- ``lapack-order``: order of LAPACK libraries to try detecting,
- ``use-ilp64``: whether to use the ILP64 interface (default: ``false``),
- ``blas-symbol-suffix``: the symbol suffix to use for the detected libraries (default: ``auto``),
- ``mkl-threading``: which MKL threading layer to use, one of ``seq``,
  ``iomp``, ``gomp``, ``tbb`` (default: ``auto``).

