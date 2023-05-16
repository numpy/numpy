.. _f2py-cmake:

===================
Using via ``cmake``
===================

In terms of complexity, ``cmake`` falls between ``make`` and ``meson``. The
learning curve is steeper since CMake syntax is not pythonic and is closer to
``make`` with environment variables.

However, the trade-off is enhanced flexibility and support for most architectures
and compilers. An introduction to the syntax is out of scope for this document,
but this `extensive CMake collection`_ of resources is great.

.. note::

   ``cmake`` is very popular for mixed-language systems, however support for
   ``f2py`` is not particularly native or pleasant; and a more natural approach
   is to consider :ref:`f2py-skbuild`

Fibonacci walkthrough (F77)
===========================

Returning to the ``fib``  example from :ref:`f2py-getting-started` section.

.. literalinclude:: ./../code/fib1.f
    :language: fortran

We do not need to explicitly generate the ``python -m numpy.f2py fib1.f``
output, which is ``fib1module.c``, which is beneficial. With this; we can now
initialize a ``CMakeLists.txt`` file as follows:

.. literalinclude:: ./../code/CMakeLists.txt
    :language: cmake

A key element of the ``CMakeLists.txt`` file defined above is that the
``add_custom_command`` is used to generate the wrapper ``C`` files and then
added as a dependency of the actual shared library target via a
``add_custom_target`` directive which prevents the command from running every
time. Additionally, the method used for obtaining the ``fortranobject.c`` file
can also be used to grab the ``numpy`` headers on older ``cmake`` versions.

This then works in the same manner as the other modules, although the naming
conventions are different and the output library is not automatically prefixed
with the ``cython`` information.

.. code:: bash

    ls .
    # CMakeLists.txt fib1.f
    cmake -S . -B build
    cmake --build build
    cd build
    python -c "import numpy as np; import fibby; a = np.zeros(9); fibby.fib(a); print (a)"
    # [ 0.  1.  1.  2.  3.  5.  8. 13. 21.]

This is particularly useful where an existing toolchain already exists and
``scikit-build`` or other additional ``python`` dependencies are discouraged.

.. _extensive CMake collection: https://cliutils.gitlab.io/modern-cmake/
