.. _f2py-meson-distutils:


1 Migrating to ``meson``
------------------------

As per the timeline laid out in :ref:`distutils-status-migration`,
``distutils`` has been removed. This page collects common workflows.

.. note::

    This is a **living** document, `pull requests <https://numpy.org/doc/stable/dev/howto-docs.html>`_ are very welcome!

1.1 Baseline
~~~~~~~~~~~~

We will start out with a slightly modern variation of the classic Fibonnaci
series generator.

.. code:: fortran

    ! fib.f90
    subroutine fib(a, n)
      use iso_c_binding
       integer(c_int), intent(in) :: n
       integer(c_int), intent(out) :: a(n)
       do i = 1, n
          if (i .eq. 1) then
             a(i) = 0.0d0
          elseif (i .eq. 2) then
             a(i) = 1.0d0
          else
             a(i) = a(i - 1) + a(i - 2)
          end if
       end do
    end

This will not win any awards, but can be a reasonable starting point.

1.2 Compilation options
~~~~~~~~~~~~~~~~~~~~~~~

1.2.1 Basic Usage
^^^^^^^^^^^^^^^^^

.. code:: bash

    python -m numpy.f2py -c fib.f90 -m fib
    ❯ python -c "import fib; print(fib.fib(30))"
    [     0      1      1      2      3      5      8     13     21     34
         55     89    144    233    377    610    987   1597   2584   4181
       6765  10946  17711  28657  46368  75025 121393 196418 317811 514229]

1.2.2 Specify the backend
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

  python -m numpy.f2py -c fib.f90 -m fib

This is the only option. There used to be a ``distutils`` backend but it was
removed in NumPy2.5.0.

1.2.3 Pass a compiler name
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

  FC=gfortran python -m numpy.f2py -c fib.f90 -m fib

Native files can also be used.

Similarly, ``CC`` can be used in both cases to set the ``C`` compiler. Since the
environment variables are generally pretty common across both, so a small
sample is included below.

.. table::

    +------------------------------------+-------------------------------+
    | **Name**                           | **What**                      |
    +------------------------------------+-------------------------------+
    | FC                                 | Fortran compiler              |
    +------------------------------------+-------------------------------+
    | CC                                 | C compiler                    |
    +------------------------------------+-------------------------------+
    | CFLAGS                             | C compiler options            |
    +------------------------------------+-------------------------------+
    | FFLAGS                             | Fortran compiler options      |
    +------------------------------------+-------------------------------+
    | LDFLAGS                            | Linker options                |
    +------------------------------------+-------------------------------+
    | LD_LIBRARY_PATH                    | Library file locations (Unix) |
    +------------------------------------+-------------------------------+
    | LIBS                               | Libraries to link against     |
    +------------------------------------+-------------------------------+
    | PATH                               | Search path for executables   |
    +------------------------------------+-------------------------------+
    | CXX                                | C++ compiler                  |
    +------------------------------------+-------------------------------+
    | CXXFLAGS                           | C++ compiler options          |
    +------------------------------------+-------------------------------+


.. note::

    For Windows, these may not work very reliably, so `native files <https://mesonbuild.com/Native-environments.html>`_ are likely the
    best bet, or by direct `1.3 Customizing builds`_.

1.2.4 Dependencies
^^^^^^^^^^^^^^^^^^

.. code-block:: bash

  python -m numpy.f2py -c fib.f90 -m fib --dep lapack

This maps to ``dependency("lapack")`` and so can be used for a wide variety
of dependencies. They can be `customized further <https://mesonbuild.com/Dependencies.html>`_
to use CMake or other systems to resolve dependencies.

1.2.5 Libraries
^^^^^^^^^^^^^^^

``meson`` is capable of linking against libraries.

.. code-block:: bash

  python -m numpy.f2py -c fib.f90 -m fib -lmylib -L/path/to/mylib

1.3 Customizing builds
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  python -m numpy.f2py -c fib.f90 -m fib --build-dir blah

The resulting build can be customized via the
`Meson Build How-To Guide <https://mesonbuild.com/howtox.html>`_.
In fact, the resulting set of files can even be committed directly and used
as a meson subproject in a separate codebase.
