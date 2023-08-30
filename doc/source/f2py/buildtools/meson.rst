.. _f2py-meson:

===================
Using via ``meson``
===================

The key advantage gained by leveraging ``meson`` over the techniques described
in :ref:`f2py-distutils` is that this feeds into existing systems and larger
projects with ease. ``meson`` has a rather pythonic syntax which makes it more
comfortable and amenable to extension for ``python`` users.

.. note::

    Meson needs to be at-least ``0.46.0`` in order to resolve the ``python`` include directories.


Fibonacci Walkthrough (F77)
===========================


We will need the generated ``C`` wrapper before we can use a general purpose
build system like ``meson``. We will acquire this by:

.. code-block:: bash

    python -m numpy.f2py fib1.f -m fib2

Now, consider the following ``meson.build`` file for the ``fib`` and ``scalar``
examples from :ref:`f2py-getting-started` section:

.. literalinclude:: ../code/meson.build

At this point the build will complete, but the import will fail:

.. code-block:: bash

   meson setup builddir
   meson compile -C builddir
   cd builddir
   python -c 'import fib2'
   Traceback (most recent call last):
   File "<string>", line 1, in <module>
   ImportError: fib2.cpython-39-x86_64-linux-gnu.so: undefined symbol: FIB_
   # Check this isn't a false positive
   nm -A fib2.cpython-39-x86_64-linux-gnu.so | grep FIB_
   fib2.cpython-39-x86_64-linux-gnu.so: U FIB_

Recall that the original example, as reproduced below, was in SCREAMCASE:

.. literalinclude:: ./../code/fib1.f
   :language: fortran

With the standard approach, the subroutine exposed to ``python`` is ``fib`` and
not ``FIB``. This means we have a few options. One approach (where possible) is
to lowercase the original Fortran file with say:

.. code-block:: bash

   tr "[:upper:]" "[:lower:]" < fib1.f > fib1.f
   python -m numpy.f2py fib1.f -m fib2
   meson --wipe builddir
   meson compile -C builddir
   cd builddir
   python -c 'import fib2'

However this requires the ability to modify the source which is not always
possible. The easiest way to solve this is to let ``f2py`` deal with it:

.. code-block:: bash

   python -m numpy.f2py fib1.f -m fib2 --lower
   meson --wipe builddir
   meson compile -C builddir
   cd builddir
   python -c 'import fib2'


Automating wrapper generation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A major pain point in the workflow defined above, is the manual tracking of
inputs. Although it would require more effort to figure out the actual outputs
for reasons discussed in :ref:`f2py-bldsys`.

.. note::

   From NumPy ``1.22.4`` onwards, ``f2py`` will deterministically generate
   wrapper files based on the input file Fortran standard (F77 or greater).
   ``--skip-empty-wrappers`` can be passed to ``f2py`` to restore the previous
   behaviour of only generating wrappers when needed by the input .

However, we can augment our workflow in a straightforward to take into account
files for which the outputs are known when the build system is set up.

.. literalinclude:: ../code/meson_upd.build

This can be compiled and run as before.

.. code-block:: bash

    rm -rf builddir
    meson setup builddir
    meson compile -C builddir
    cd builddir
    python -c "import numpy as np; import fibby; a = np.zeros(9); fibby.fib(a); print (a)"
    # [ 0.  1.  1.  2.  3.  5.  8. 13. 21.]

Salient points
===============

It is worth keeping in mind the following:

* ``meson`` will default to passing ``-fimplicit-none`` under ``gfortran`` by
  default, which differs from that of the standard ``np.distutils`` behaviour

* It is not possible to use SCREAMCASE in this context, so either the contents
  of the ``.f`` file or the generated wrapper ``.c`` needs to be lowered to
  regular letters; which can be facilitated by the ``--lower`` option of
  ``F2PY``
