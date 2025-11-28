.. _f2py-getting-started:

======================================
 Three ways to wrap - getting started
======================================

Wrapping Fortran or C functions to Python using F2PY consists of the
following steps:

* Creating the so-called :doc:`signature file <signature-file>` that contains
  descriptions of wrappers to Fortran or C functions, also called the signatures
  of the functions. For Fortran routines, F2PY can create an initial signature
  file by scanning Fortran source codes and tracking all relevant information
  needed to create wrapper functions.

  * Optionally, F2PY-created signature files can be edited to optimize wrapper
    functions, which can make them "smarter" and more "Pythonic".

* F2PY reads a signature file and writes a Python C/API module containing
  Fortran/C/Python bindings.

* F2PY compiles all sources and builds an extension module containing
  the wrappers.

  * In building the extension modules, F2PY uses ``meson`` and used to use
    ``numpy.distutils`` For different build systems, see :ref:`f2py-bldsys`.


.. note::

   See :ref:`f2py-meson-distutils` for migration information.


  * Depending on your operating system, you may need to install the Python
    development headers (which provide the file ``Python.h``) separately. In
    Linux Debian-based distributions this package should be called ``python3-dev``,
    in Fedora-based distributions it is ``python3-devel``. For macOS, depending
    how Python was installed, your mileage may vary. In Windows, the headers are
    typically installed already, see :ref:`f2py-windows`.

.. note::

   F2PY supports all the operating systems SciPy is tested on so their
   `system dependencies panel`_ is a good reference.

Depending on the situation, these steps can be carried out in a single composite
command or step-by-step; in which case some steps can be omitted or combined
with others.

Below, we describe three typical approaches of using F2PY with Fortran 77. These
can be read in order of increasing effort, but also cater to different access
levels depending on whether the Fortran code can be freely modified.

The following example Fortran 77 code will be used for
illustration, save it as ``fib1.f``:

.. literalinclude:: ./code/fib1.f
   :language: fortran

.. note::

  F2PY parses Fortran/C signatures to build wrapper functions to be used with
  Python. However, it is not a compiler, and does not check for additional
  errors in source code, nor does it implement the entire language standards.
  Some errors may pass silently (or as warnings) and need to be verified by the
  user.

The quick way
==============

The quickest way to wrap the Fortran subroutine ``FIB`` for use in Python is to
run

::

  python -m numpy.f2py -c fib1.f -m fib1

or, alternatively, if the ``f2py`` command-line tool is available,

::

  f2py -c fib1.f -m fib1

.. note::

  Because the ``f2py`` command might not be available in all system, notably on
  Windows, we will use the ``python -m numpy.f2py`` command throughout this
  guide.

This command compiles and wraps ``fib1.f`` (``-c``) to create the extension
module ``fib1.so`` (``-m``) in the current directory. A list of command line
options can be seen by executing ``python -m numpy.f2py``.  Now, in Python the
Fortran subroutine ``FIB`` is accessible via ``fib1.fib``::

  >>> import numpy as np
  >>> import fib1
  >>> print(fib1.fib.__doc__)
  fib(a,[n])

  Wrapper for ``fib``.

  Parameters
  ----------
  a : input rank-1 array('d') with bounds (n)

  Other parameters
  ----------------
  n : input int, optional
      Default: len(a)

  >>> a = np.zeros(8, 'd')
  >>> fib1.fib(a)
  >>> print(a)
  [  0.   1.   1.   2.   3.   5.   8.  13.]

.. note::

  * Note that F2PY recognized that the second argument ``n`` is the
    dimension of the first array argument ``a``. Since by default all
    arguments are input-only arguments, F2PY concludes that ``n`` can
    be optional with the default value ``len(a)``.

  * One can use different values for optional ``n``::

      >>> a1 = np.zeros(8, 'd')
      >>> fib1.fib(a1, 6)
      >>> print(a1)
      [ 0.  1.  1.  2.  3.  5.  0.  0.]

    but an exception is raised when it is incompatible with the input
    array ``a``::

      >>> fib1.fib(a, 10)
      Traceback (most recent call last):
        File "<stdin>", line 1, in <module>
      fib.error: (len(a)>=n) failed for 1st keyword n: fib:n=10
      >>>

    F2PY implements basic compatibility checks between related
    arguments in order to avoid unexpected crashes.

  * When a NumPy array that is :term:`Fortran <Fortran order>`
    :term:`contiguous` and has a ``dtype`` corresponding to a presumed Fortran
    type is used as an input array argument, then its C pointer is directly
    passed to Fortran.

    Otherwise, F2PY makes a contiguous copy (with the proper ``dtype``) of the
    input array and passes a C pointer of the copy to the Fortran subroutine. As
    a result, any possible changes to the (copy of) input array have no effect
    on the original argument, as demonstrated below::

      >>> a = np.ones(8, 'i')
      >>> fib1.fib(a)
      >>> print(a)
      [1 1 1 1 1 1 1 1]

    Clearly, this is unexpected, as Fortran typically passes by reference. That
    the above example worked with ``dtype=float`` is considered accidental.

    F2PY provides an ``intent(inplace)`` attribute that modifies the attributes
    of an input array so that any changes made by the Fortran routine will be
    reflected in the input argument. For example, if one specifies the
    ``intent(inplace) a`` directive (see :ref:`f2py-attributes` for details),
    then the example above would read::

      >>> a = np.ones(8, 'i')
      >>> fib1.fib(a)
      >>> print(a)
      [  0.   1.   1.   2.   3.   5.   8.  13.]

    However, the recommended way to have changes made by Fortran subroutine
    propagate to Python is to use the ``intent(out)`` attribute. That approach
    is more efficient and also cleaner.

  * The usage of ``fib1.fib`` in Python is very similar to using ``FIB`` in
    Fortran. However, using *in situ* output arguments in Python is poor style,
    as there are no safety mechanisms in Python to protect against wrong
    argument types. When using Fortran or C, compilers discover any type
    mismatches during the compilation process, but in Python the types must be
    checked at runtime. Consequently, using *in situ* output arguments in Python
    may lead to difficult to find bugs, not to mention the fact that the
    codes will be less readable when all required type checks are implemented.

  Though the approach to wrapping Fortran routines for Python discussed so far
  is very straightforward, it has several drawbacks (see the comments above).
  The drawbacks are due to the fact that there is no way for F2PY to determine
  the actual intention of the arguments; that is, there is ambiguity in
  distinguishing between input and output arguments. Consequently, F2PY assumes
  that all arguments are input arguments by default.

  There are ways (see below) to remove this ambiguity by "teaching" F2PY about
  the true intentions of function arguments, and F2PY is then able to generate
  more explicit, easier to use, and less error prone wrappers for Fortran
  functions.

The smart way
==============

If we want to have more control over how F2PY will treat the interface to our
Fortran code, we can apply the wrapping steps one by one.

* First, we create a signature file from ``fib1.f`` by running:

  ::

    python -m numpy.f2py fib1.f -m fib2 -h fib1.pyf

  The signature file is saved to ``fib1.pyf`` (see the ``-h`` flag) and its
  contents are shown below.

  .. literalinclude:: ./code/fib1.pyf
     :language: fortran

* Next, we'll teach F2PY that the argument ``n`` is an input argument (using the
  ``intent(in)`` attribute) and that the result, i.e., the contents of ``a``
  after calling the Fortran function ``FIB``, should be returned to Python
  (using the ``intent(out)`` attribute). In addition, an array ``a`` should be
  created dynamically using the size determined by the input argument ``n``
  (using the ``depend(n)`` attribute to indicate this dependence relation).

  The contents of a suitably modified version of ``fib1.pyf`` (saved as
  ``fib2.pyf``) are as follows:

  .. literalinclude:: ./code/fib2.pyf
     :language: fortran

* Finally, we build the extension module with ``numpy.distutils`` by running:

  ::

    python -m numpy.f2py -c fib2.pyf fib1.f

In Python::

  >>> import fib2
  >>> print(fib2.fib.__doc__)
  a = fib(n)

  Wrapper for ``fib``.

  Parameters
  ----------
  n : input int

  Returns
  -------
  a : rank-1 array('d') with bounds (n)

  >>> print(fib2.fib(8))
  [  0.   1.   1.   2.   3.   5.   8.  13.]

.. note::

  * The signature of ``fib2.fib`` now more closely corresponds to the intention
    of the Fortran subroutine ``FIB``: given the number ``n``, ``fib2.fib``
    returns the first ``n`` Fibonacci numbers as a NumPy array. The new Python
    signature ``fib2.fib`` also rules out the unexpected behaviour in
    ``fib1.fib``.

  * Note that by default, using a single ``intent(out)`` also implies
    ``intent(hide)``. Arguments that have the ``intent(hide)`` attribute
    specified will not be listed in the argument list of a wrapper function.

  For more details, see :doc:`signature-file`.

The quick and smart way
========================

The "smart way" of wrapping Fortran functions, as explained above, is
suitable for wrapping (e.g. third party) Fortran codes for which
modifications to their source codes are not desirable nor even
possible.

However, if editing Fortran codes is acceptable, then the generation of an
intermediate signature file can be skipped in most cases. F2PY specific
attributes can be inserted directly into Fortran source codes using F2PY
directives. A F2PY directive consists of special comment lines (starting with
``Cf2py`` or ``!f2py``, for example) which are ignored by Fortran compilers but
interpreted by F2PY as normal lines.

Consider a modified version of the previous Fortran code with F2PY directives,
saved as ``fib3.f``:

.. literalinclude:: ./code/fib3.f
   :language: fortran

Building the extension module can be now carried out in one command::

  python -m numpy.f2py -c -m fib3 fib3.f

Notice that the resulting wrapper to ``FIB`` is as "smart" (unambiguous) as in
the previous case::

  >>> import fib3
  >>> print(fib3.fib.__doc__)
  a = fib(n)

  Wrapper for ``fib``.

  Parameters
  ----------
  n : input int

  Returns
  -------
  a : rank-1 array('d') with bounds (n)

  >>> print(fib3.fib(8))
  [  0.   1.   1.   2.   3.   5.   8.  13.]

.. _`system dependencies panel`: https://scipy.github.io/devdocs/building/index.html#system-level-dependencies
