.. _f2py-meson-distutils:


1 Migrating to ``meson``
------------------------

As per the timeline laid out in :ref:`distutils-status-migration`,
``distutils`` has ceased to be the default build backend for ``f2py``. This page
collects common workflows in both formats.

.. note::

    This is a ****living**** document, `pull requests <https://numpy.org/doc/stable/dev/howto-docs.html>`_ are very welcome!

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

This is unchanged:

.. code:: bash

    python -m numpy.f2py -c fib.f90 -m fib
    ❯ python -c "import fib; print(fib.fib(30))"
    [     0      1      1      2      3      5      8     13     21     34
         55     89    144    233    377    610    987   1597   2584   4181
       6765  10946  17711  28657  46368  75025 121393 196418 317811 514229]

1.2.2 Specify the backend
^^^^^^^^^^^^^^^^^^^^^^^^^

.. tab-set::

  .. tab-item:: Distutils
    :sync: distutils

    .. code-block:: bash

      python -m numpy.f2py -c fib.f90 -m fib --backend distutils

    This is the default for Python versions before 3.12.

  .. tab-item:: Meson
    :sync: meson

    .. code-block:: bash

      python -m numpy.f2py -c fib.f90 -m fib --backend meson

    This is the only option for Python versions after 3.12.

1.2.3 Pass a compiler name
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. tab-set::

  .. tab-item:: Distutils
    :sync: distutils

    .. code-block:: bash

      python -m numpy.f2py -c fib.f90 -m fib --backend distutils --fcompiler=gfortran

  .. tab-item:: Meson
    :sync: meson

    .. code-block:: bash

      FC="gfortran" python -m numpy.f2py -c fib.f90 -m fib --backend meson

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
    | LD\ :sub:`LIBRARY`\ \ :sub:`PATH`\ | Library file locations (Unix) |
    +------------------------------------+-------------------------------+
    | LIBS                               | Libraries to link against     |
    +------------------------------------+-------------------------------+
    | PATH                               | Search path for executables   |
    +------------------------------------+-------------------------------+
    | LDFLAGS                            | Linker flags                  |
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

Here, ``meson`` can actually be used to set dependencies more robustly.

.. tab-set::

  .. tab-item:: Distutils
    :sync: distutils

    .. code-block:: bash

      python -m numpy.f2py -c fib.f90 -m fib --backend distutils -llapack

    Note that this approach in practice is error prone.

  .. tab-item:: Meson
    :sync: meson

    .. code-block:: bash

      python -m numpy.f2py -c fib.f90 -m fib --backend meson --dep lapack

    This maps to ``dependency("lapack")`` and so can be used for a wide variety
    of dependencies. They can be `customized further <https://mesonbuild.com/Dependencies.html>`_
    to use CMake or other systems to resolve dependencies.

1.2.5 Libraries
^^^^^^^^^^^^^^^

Both ``meson`` and ``distutils`` are capable of linking against libraries.

.. tab-set::

  .. tab-item:: Distutils
    :sync: distutils

    .. code-block:: bash

      python -m numpy.f2py -c fib.f90 -m fib --backend distutils -lmylib -L/path/to/mylib

  .. tab-item:: Meson
    :sync: meson

    .. code-block:: bash

      python -m numpy.f2py -c fib.f90 -m fib --backend meson -lmylib -L/path/to/mylib

1.3 Customizing builds
~~~~~~~~~~~~~~~~~~~~~~

.. tab-set::

  .. tab-item:: Distutils
    :sync: distutils

    .. code-block:: bash

      python -m numpy.f2py -c fib.f90 -m fib --backend distutils --build-dir blah

    This can be technically integrated with other codes, see :ref:`f2py-distutils`.

  .. tab-item:: Meson
    :sync: meson

    .. code-block:: bash

      python -m numpy.f2py -c fib.f90 -m fib --backend meson --build-dir blah

    The resulting build can be customized via the
    `Meson Build How-To Guide <https://mesonbuild.com/howtox.html>`_.
    In fact, the resulting set of files can even be commited directly and used
    as a meson subproject in a separate codebase.
