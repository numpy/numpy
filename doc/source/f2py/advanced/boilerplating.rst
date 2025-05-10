.. _f2py-boilerplating:

====================================
Boilerplate reduction and templating
====================================

Using FYPP for binding generic interfaces
=========================================

``f2py`` doesn't currently support binding interface blocks. However, there are
workarounds in use. Perhaps the best known is the usage of ``tempita`` for using
``.pyf.src`` files as is done in the bindings which `are part of scipy`_. `tempita` support has been removed and
is no longer recommended in any case.

.. note::
    The reason interfaces cannot be supported within ``f2py`` itself is because
    they don't correspond to exported symbols in compiled libraries.

    .. code:: sh

       ❯ nm gen.o
        0000000000000078 T __add_mod_MOD_add_complex
        0000000000000000 T __add_mod_MOD_add_complex_dp
        0000000000000150 T __add_mod_MOD_add_integer
        0000000000000124 T __add_mod_MOD_add_real
        00000000000000ee T __add_mod_MOD_add_real_dp

Here we will discuss a few techniques to leverage ``f2py`` in conjunction with
`fypp`_ to emulate generic interfaces and to ease the binding of multiple
(similar) functions.


Basic example: Addition module
------------------------------

Let us build on the example (from the user guide, :ref:`f2py-examples`) of a
subroutine which takes in two arrays and returns its sum.

.. literalinclude:: ./../code/add.f
    :language: fortran


We will recast this into modern fortran:

.. literalinclude:: ./../code/advanced/boilerplating/src/adder_base.f90
    :language: fortran

We could go on as in the original example, adding intents by hand among other
things, however in production often there are other concerns. For one, we can
template via FYPP the construction of similar functions:

.. literalinclude:: ./../code/advanced/boilerplating/src/gen_adder.f90.fypp

This can be pre-processed to generate the full fortran code:

.. code:: sh

       ❯ fypp gen_adder.f90.fypp > adder.f90

As to be expected, this can be wrapped by ``f2py`` subsequently.

Now we will consider maintaining the bindings in a separate file. Note the
following basic ``.pyf`` which can be generated for a single subroutine via ``f2py -m adder adder_base.f90 -h adder.pyf``:

.. literalinclude:: ./../code/advanced/boilerplating/src/base_adder.pyf
    :language: fortran

With the docstring:

.. literalinclude:: ./../code/advanced/boilerplating/res/base_docstring.dat
    :language: reST

Which is already pretty good. However, ``n`` should never be passed in the first
place so we will make some minor adjustments.

.. literalinclude:: ./../code/advanced/boilerplating/src/improved_base_adder.pyf
    :language: fortran

Which corresponds to:

.. literalinclude:: ./../code/advanced/boilerplating/res/improved_docstring.dat
    :language: reST

Finally, we can template over this in a similar manner, to attain the original
goal of having bindings which make use of ``f2py`` directives and have minimal
spurious repetition.

.. literalinclude:: ./../code/advanced/boilerplating/src/adder.pyf.fypp

Usage boils down to:

.. code:: sh

   fypp gen_adder.f90.fypp > adder.f90
   fypp adder.pyf.fypp > adder.pyf
   f2py -m adder -c adder.pyf adder.f90 --backend meson

.. _`fypp`: https://fypp.readthedocs.io/en/stable/fypp.html
.. _`are part of scipy`: https://github.com/scipy/scipy/blob/c93da6f46dbed8b3cc0ccd2495b5678f7b740a03/scipy/linalg/clapack.pyf.src
