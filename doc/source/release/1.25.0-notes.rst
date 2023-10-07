.. currentmodule:: numpy

==========================
NumPy 1.25.0 Release Notes
==========================

The NumPy 1.25.0 release continues the ongoing work to improve the handling and
promotion of dtypes, increase the execution speed, and clarify the
documentation. There has also been work to prepare for the future NumPy 2.0.0
release, resulting in a large number of new and expired deprecation.
Highlights are:

- Support for MUSL, there are now MUSL wheels.
- Support the Fujitsu C/C++ compiler.
- Object arrays are now supported in einsum
- Support for inplace matrix multiplication (``@=``).

We will be releasing a NumPy 1.26 when Python 3.12 comes out. That is needed
because distutils has been dropped by Python 3.12 and we will be switching to using
meson for future builds. The next mainline release will be NumPy 2.0.0. We plan
that the 2.0 series will still support downstream projects built against earlier
versions of NumPy.

The Python versions supported in this release are 3.9-3.11.


Deprecations
============

* ``np.core.MachAr`` is deprecated.  It is private API.  In names
  defined in ``np.core`` should generally be considered private.

  (`gh-22638 <https://github.com/numpy/numpy/pull/22638>`__)

* ``np.finfo(None)`` is deprecated.

  (`gh-23011 <https://github.com/numpy/numpy/pull/23011>`__)

* ``np.round_`` is deprecated. Use `np.round` instead.

  (`gh-23302 <https://github.com/numpy/numpy/pull/23302>`__)

* ``np.product`` is deprecated. Use `np.prod` instead.

  (`gh-23314 <https://github.com/numpy/numpy/pull/23314>`__)

* ``np.cumproduct`` is deprecated. Use `np.cumprod` instead.

  (`gh-23314 <https://github.com/numpy/numpy/pull/23314>`__)

* ``np.sometrue`` is deprecated. Use `np.any` instead.

  (`gh-23314 <https://github.com/numpy/numpy/pull/23314>`__)

* ``np.alltrue`` is deprecated. Use `np.all` instead.

  (`gh-23314 <https://github.com/numpy/numpy/pull/23314>`__)

* Only ndim-0 arrays are treated as scalars.  NumPy used to treat all arrays of
  size 1 (e.g., ``np.array([3.14])``) as scalars.  In the future, this will be
  limited to arrays of ndim 0 (e.g., ``np.array(3.14)``).  The following
  expressions will report a deprecation warning:

  .. code-block:: python

      a = np.array([3.14])
      float(a)  # better: a[0] to get the numpy.float or a.item()

      b = np.array([[3.14]])
      c = numpy.random.rand(10)
      c[0] = b  # better: c[0] = b[0, 0]

  (`gh-10615 <https://github.com/numpy/numpy/pull/10615>`__)

* ``np.find_common_type`` is deprecated.
  `numpy.find_common_type` is now deprecated and its use should be replaced
  with either `numpy.result_type` or `numpy.promote_types`.
  Most users leave the second ``scalar_types`` argument to ``find_common_type``
  as ``[]`` in which case ``np.result_type`` and ``np.promote_types`` are both
  faster and more robust.
  When not using ``scalar_types`` the main difference is that the replacement
  intentionally converts non-native byte-order to native byte order.
  Further, ``find_common_type`` returns ``object`` dtype rather than failing
  promotion.  This leads to differences when the inputs are not all numeric.
  Importantly, this also happens for e.g. timedelta/datetime for which NumPy
  promotion rules are currently sometimes surprising.

  When the ``scalar_types`` argument is not ``[]`` things are more complicated.
  In most cases, using ``np.result_type`` and passing the Python values
  ``0``, ``0.0``, or ``0j`` has the same result as using ``int``, ``float``,
  or ``complex`` in `scalar_types`.

  When ``scalar_types`` is constructed, ``np.result_type`` is the
  correct replacement and it may be passed scalar values like ``np.float32(0.0)``.
  Passing values other than 0, may lead to value-inspecting behavior
  (which ``np.find_common_type`` never used and NEP 50 may change in the future).
  The main possible change in behavior in this case, is when the array types
  are signed integers and scalar types are unsigned.

  If you are unsure about how to replace a use of ``scalar_types`` or when
  non-numeric dtypes are likely, please do not hesitate to open a NumPy issue
  to ask for help.

  (`gh-22539 <https://github.com/numpy/numpy/pull/22539>`__)


Expired deprecations
====================

* ``np.core.machar`` and ``np.finfo.machar`` have been removed.

  (`gh-22638 <https://github.com/numpy/numpy/pull/22638>`__)

* ``+arr`` will now raise an error when the dtype is not
  numeric (and positive is undefined).

  (`gh-22998 <https://github.com/numpy/numpy/pull/22998>`__)

* A sequence must now be passed into the stacking family of functions
  (``stack``, ``vstack``, ``hstack``, ``dstack`` and ``column_stack``).

  (`gh-23019 <https://github.com/numpy/numpy/pull/23019>`__)

* ``np.clip`` now defaults to same-kind casting. Falling back to
  unsafe casting was deprecated in NumPy 1.17.

  (`gh-23403 <https://github.com/numpy/numpy/pull/23403>`__)

* ``np.clip`` will now propagate ``np.nan`` values passed as ``min`` or ``max``.
  Previously, a scalar NaN was usually ignored.  This was deprecated in NumPy 1.17.

  (`gh-23403 <https://github.com/numpy/numpy/pull/23403>`__)

* The ``np.dual`` submodule has been removed.

  (`gh-23480 <https://github.com/numpy/numpy/pull/23480>`__)

* NumPy now always ignores sequence behavior for an array-like (defining
  one of the array protocols).  (Deprecation started NumPy 1.20)

  (`gh-23660 <https://github.com/numpy/numpy/pull/23660>`__)

* The niche ``FutureWarning`` when casting to a subarray dtype in ``astype``
  or the array creation functions such as ``asarray`` is now finalized.
  The behavior is now always the same as if the subarray dtype was
  wrapped into a single field (which was the workaround, previously).
  (FutureWarning since NumPy 1.20)

  (`gh-23666 <https://github.com/numpy/numpy/pull/23666>`__)

* ``==`` and ``!=`` warnings have been finalized.  The ``==`` and ``!=``
  operators on arrays now always:

  * raise errors that occur during comparisons such as when the arrays
    have incompatible shapes (``np.array([1, 2]) == np.array([1, 2, 3])``).
  * return an array of all ``True`` or all ``False`` when values are
    fundamentally not comparable (e.g. have different dtypes).  An example
    is ``np.array(["a"]) == np.array([1])``.

    This mimics the Python behavior of returning ``False`` and ``True``
    when comparing incompatible types like ``"a" == 1`` and ``"a" != 1``.
    For a long time these gave ``DeprecationWarning`` or ``FutureWarning``.

  (`gh-22707 <https://github.com/numpy/numpy/pull/22707>`__)

* Nose support has been removed. NumPy switched to using pytest in 2018 and nose
  has been unmaintained for many years. We have kept NumPy's nose support to
  avoid breaking downstream projects who might have been using it and not yet
  switched to pytest or some other testing framework. With the arrival of
  Python 3.12, unpatched nose will raise an error. It is time to move on.

  *Decorators removed*:

  - raises
  - slow
  - setastest
  - skipif
  - knownfailif
  - deprecated
  - parametrize
  - _needs_refcount

  These are not to be confused with pytest versions with similar names, e.g.,
  pytest.mark.slow, pytest.mark.skipif, pytest.mark.parametrize.

  *Functions removed*:

  - Tester
  - import_nose
  - run_module_suite

  (`gh-23041 <https://github.com/numpy/numpy/pull/23041>`__)

* The ``numpy.testing.utils`` shim has been removed.  Importing from the
  ``numpy.testing.utils`` shim has been deprecated since 2019, the shim has now
  been removed. All imports should be made directly from ``numpy.testing``.

  (`gh-23060 <https://github.com/numpy/numpy/pull/23060>`__)

* The environment variable to disable dispatching has been removed.
  Support for the ``NUMPY_EXPERIMENTAL_ARRAY_FUNCTION`` environment variable has
  been removed. This variable disabled dispatching with ``__array_function__``.

  (`gh-23376 <https://github.com/numpy/numpy/pull/23376>`__)

* Support for ``y=`` as an alias of ``out=`` has been removed.
  The ``fix``, ``isposinf`` and ``isneginf`` functions allowed using ``y=`` as a
  (deprecated) alias for ``out=``. This is no longer supported.

  (`gh-23376 <https://github.com/numpy/numpy/pull/23376>`__)


Compatibility notes
===================

* The ``busday_count`` method now correctly handles cases where the ``begindates`` is later in time
  than the ``enddates``. Previously, the ``enddates`` was included, even though the documentation states
  it is always excluded.

  (`gh-23229 <https://github.com/numpy/numpy/pull/23229>`__)

* When comparing datetimes and timedelta using ``np.equal`` or ``np.not_equal``
  numpy previously allowed the comparison with ``casting="unsafe"``.
  This operation now fails. Forcing the output dtype using the ``dtype``
  kwarg can make the operation succeed, but we do not recommend it.

  (`gh-22707 <https://github.com/numpy/numpy/pull/22707>`__)

* When loading data from a file handle using ``np.load``,
  if the handle is at the end of file, as can happen when reading
  multiple arrays by calling ``np.load`` repeatedly, numpy previously
  raised ``ValueError`` if ``allow_pickle=False``, and ``OSError`` if
  ``allow_pickle=True``. Now it raises ``EOFError`` instead, in both cases.

  (`gh-23105 <https://github.com/numpy/numpy/pull/23105>`__)

``np.pad`` with ``mode=wrap`` pads with strict multiples of original data
-------------------------------------------------------------------------
Code based on earlier version of ``pad`` that uses  ``mode="wrap"`` will return
different results when the padding size is larger than initial array.

``np.pad`` with ``mode=wrap`` now always fills the space with
strict multiples of original data even if the padding size is larger than the
initial array.

(`gh-22575 <https://github.com/numpy/numpy/pull/22575>`__)

Cython ``long_t`` and ``ulong_t`` removed
-----------------------------------------
``long_t`` and ``ulong_t`` were aliases for ``longlong_t`` and ``ulonglong_t``
and confusing (a remainder from of Python 2).  This change may lead to the errors::

     'long_t' is not a type identifier
     'ulong_t' is not a type identifier

We recommend use of bit-sized types such as ``cnp.int64_t`` or the use of
``cnp.intp_t`` which is 32 bits on 32 bit systems and 64 bits on 64 bit
systems (this is most compatible with indexing).
If C ``long`` is desired, use plain ``long`` or ``npy_long``.
``cnp.int_t`` is also ``long`` (NumPy's default integer).  However, ``long``
is 32 bit on 64 bit windows and we may wish to adjust this even in NumPy.
(Please do not hesitate to contact NumPy developers if you are curious about this.)

(`gh-22637 <https://github.com/numpy/numpy/pull/22637>`__)

Changed error message and type for bad ``axes`` argument to ``ufunc``
---------------------------------------------------------------------
The error message and type when a wrong ``axes`` value is passed to
``ufunc(..., axes=[...])``` has changed. The message is now more indicative of
the problem, and if the value is mismatched an ``AxisError`` will be raised.
A ``TypeError`` will still be raised for invalid input types.

(`gh-22675 <https://github.com/numpy/numpy/pull/22675>`__)

Array-likes that define ``__array_ufunc__`` can now override ufuncs if used as ``where``
----------------------------------------------------------------------------------------
If the ``where`` keyword argument of a :class:`numpy.ufunc` is a subclass of
:class:`numpy.ndarray` or is a duck type that defines
:func:`numpy.class.__array_ufunc__` it can override the behavior of the ufunc
using the same mechanism as the input and output arguments.
Note that for this to work properly, the ``where.__array_ufunc__``
implementation will have to unwrap the ``where`` argument to pass it into the
default implementation of the ``ufunc`` or, for :class:`numpy.ndarray`
subclasses before using ``super().__array_ufunc__``.

(`gh-23240 <https://github.com/numpy/numpy/pull/23240>`__)

Compiling against the NumPy C API is now backwards compatible by default
------------------------------------------------------------------------
NumPy now defaults to exposing a backwards compatible subset of the C-API.
This makes the use of ``oldest-supported-numpy`` unnecessary.
Libraries can override the default minimal version to be compatible with
using::

    #define NPY_TARGET_VERSION NPY_1_22_API_VERSION

before including NumPy or by passing the equivalent ``-D`` option to the
compiler.
The NumPy 1.25 default is ``NPY_1_19_API_VERSION``.  Because the NumPy 1.19
C API was identical to the NumPy 1.16 one resulting programs will be compatible
with NumPy 1.16 (from a C-API perspective).
This default will be increased in future non-bugfix releases.
You can still compile against an older NumPy version and run on a newer one.

For more details please see :ref:`for-downstream-package-authors`.

(`gh-23528 <https://github.com/numpy/numpy/pull/23528>`__)


New Features
============

``np.einsum`` now accepts arrays with ``object`` dtype
------------------------------------------------------
The code path will call python operators on object dtype arrays, much
like ``np.dot`` and ``np.matmul``.

(`gh-18053 <https://github.com/numpy/numpy/pull/18053>`__)

Add support for inplace matrix multiplication
---------------------------------------------
It is now possible to perform inplace matrix multiplication
via the ``@=`` operator.

.. code-block:: python

    >>> import numpy as np

    >>> a = np.arange(6).reshape(3, 2)
    >>> print(a)
    [[0 1]
     [2 3]
     [4 5]]

    >>> b = np.ones((2, 2), dtype=int)
    >>> a @= b
    >>> print(a)
    [[1 1]
     [5 5]
     [9 9]]

(`gh-21120 <https://github.com/numpy/numpy/pull/21120>`__)

Added ``NPY_ENABLE_CPU_FEATURES`` environment variable
------------------------------------------------------
Users may now choose to enable only a subset of the built CPU features at
runtime by specifying the `NPY_ENABLE_CPU_FEATURES` environment variable.
Note that these specified features must be outside the baseline, since those
are always assumed. Errors will be raised if attempting to enable a feature
that is either not supported by your CPU, or that NumPy was not built with.

(`gh-22137 <https://github.com/numpy/numpy/pull/22137>`__)

NumPy now has an ``np.exceptions`` namespace
--------------------------------------------
NumPy now has a dedicated namespace making most exceptions
and warnings available.  All of these remain available in the
main namespace, although some may be moved slowly in the future.
The main reason for this is to increase discoverability and add
future exceptions.

(`gh-22644 <https://github.com/numpy/numpy/pull/22644>`__)

``np.linalg`` functions return NamedTuples
------------------------------------------
``np.linalg`` functions that return tuples now return namedtuples. These
functions are ``eig()``, ``eigh()``, ``qr()``, ``slogdet()``, and ``svd()``.
The return type is unchanged in instances where these functions return
non-tuples with certain keyword arguments (like ``svd(compute_uv=False)``).

(`gh-22786 <https://github.com/numpy/numpy/pull/22786>`__)

String functions in ``np.char`` are compatible with NEP 42 custom dtypes
------------------------------------------------------------------------
Custom dtypes that represent unicode strings or byte strings can now be
passed to the string functions in ``np.char``.

(`gh-22863 <https://github.com/numpy/numpy/pull/22863>`__)

String dtype instances can be created from the string abstract dtype classes
----------------------------------------------------------------------------
It is now possible to create a string dtype instance with a size without
using the string name of the dtype. For example, ``type(np.dtype('U'))(8)``
will create a dtype that is equivalent to ``np.dtype('U8')``. This feature
is most useful when writing generic code dealing with string dtype
classes.

(`gh-22963 <https://github.com/numpy/numpy/pull/22963>`__)

Fujitsu C/C++ compiler is now supported
---------------------------------------
Support for Fujitsu compiler has been added.
To build with Fujitsu compiler, run:

    python setup.py build -c fujitsu


SSL2 is now supported
---------------------
Support for SSL2 has been added. SSL2 is a library that provides OpenBLAS
compatible GEMM functions.  To enable SSL2, it need to edit site.cfg and build
with Fujitsu compiler.  See site.cfg.example.

(`gh-22982 <https://github.com/numpy/numpy/pull/22982>`__)


Improvements
============

``NDArrayOperatorsMixin`` specifies that it has no ``__slots__``
----------------------------------------------------------------
The ``NDArrayOperatorsMixin`` class now specifies that it contains no
``__slots__``, ensuring that subclasses can now make use of this feature in
Python.

(`gh-23113 <https://github.com/numpy/numpy/pull/23113>`__)

Fix power of complex zero
-------------------------
``np.power`` now returns a different result for ``0^{non-zero}``
for complex numbers.  Note that the value is only defined when
the real part of the exponent is larger than zero.
Previously, NaN was returned unless the imaginary part was strictly
zero.  The return value is either ``0+0j`` or ``0-0j``.

(`gh-18535 <https://github.com/numpy/numpy/pull/18535>`__)

New ``DTypePromotionError``
---------------------------
NumPy now has a new ``DTypePromotionError`` which is used when two
dtypes cannot be promoted to a common one, for example::

    np.result_type("M8[s]", np.complex128)

raises this new exception.

(`gh-22707 <https://github.com/numpy/numpy/pull/22707>`__)

`np.show_config` uses information from Meson
--------------------------------------------
Build and system information now contains information from Meson.
`np.show_config` now has a new optional parameter ``mode`` to help
customize the output.

(`gh-22769 <https://github.com/numpy/numpy/pull/22769>`__)

Fix ``np.ma.diff`` not preserving the mask when called with arguments prepend/append.
-------------------------------------------------------------------------------------
Calling ``np.ma.diff`` with arguments prepend and/or append now returns a
``MaskedArray`` with the input mask preserved.

Previously, a ``MaskedArray`` without the mask was returned.

(`gh-22776 <https://github.com/numpy/numpy/pull/22776>`__)

Corrected error handling for NumPy C-API in Cython
--------------------------------------------------
Many NumPy C functions defined for use in Cython were lacking the
correct error indicator like ``except -1`` or ``except *``.
These have now been added.

(`gh-22997 <https://github.com/numpy/numpy/pull/22997>`__)

Ability to directly spawn random number generators
--------------------------------------------------
`numpy.random.Generator.spawn` now allows to directly spawn new
independent child generators via the `numpy.random.SeedSequence.spawn`
mechanism.
`numpy.random.BitGenerator.spawn` does the same for the underlying
bit generator.

Additionally, `numpy.random.BitGenerator.seed_seq` now gives direct
access to the seed sequence used for initializing the bit generator.
This allows for example::

    seed = 0x2e09b90939db40c400f8f22dae617151
    rng = np.random.default_rng(seed)
    child_rng1, child_rng2 = rng.spawn(2)

    # safely use rng, child_rng1, and child_rng2

Previously, this was hard to do without passing the ``SeedSequence``
explicitly.  Please see `numpy.random.SeedSequence` for more information.

(`gh-23195 <https://github.com/numpy/numpy/pull/23195>`__)

``numpy.logspace`` now supports a non-scalar ``base`` argument
--------------------------------------------------------------
The ``base`` argument of ``numpy.logspace`` can now be array-like if it is
broadcastable against the ``start`` and ``stop`` arguments.

(`gh-23275 <https://github.com/numpy/numpy/pull/23275>`__)

``np.ma.dot()`` now supports for non-2d arrays
----------------------------------------------
Previously ``np.ma.dot()`` only worked if ``a`` and ``b`` were both 2d.
Now it works for non-2d arrays as well as ``np.dot()``.

(`gh-23322 <https://github.com/numpy/numpy/pull/23322>`__)

Explicitly show keys of .npz file in repr
-----------------------------------------
``NpzFile`` shows keys of loaded .npz file when printed.

.. code-block:: python

   >>> npzfile = np.load('arr.npz')
   >>> npzfile
   NpzFile 'arr.npz' with keys arr_0, arr_1, arr_2, arr_3, arr_4...

(`gh-23357 <https://github.com/numpy/numpy/pull/23357>`__)

NumPy now exposes DType classes in ``np.dtypes``
------------------------------------------------
The new ``numpy.dtypes`` module now exposes DType classes and
will contain future dtype related functionality.
Most users should have no need to use these classes directly.

(`gh-23358 <https://github.com/numpy/numpy/pull/23358>`__)

Drop dtype metadata before saving in .npy or .npz files
-------------------------------------------------------
Currently, a ``*.npy`` file containing a table with a dtype with
metadata cannot be read back.
Now, `np.save` and `np.savez` drop metadata before saving.

(`gh-23371 <https://github.com/numpy/numpy/pull/23371>`__)

``numpy.lib.recfunctions.structured_to_unstructured`` returns views in more cases
---------------------------------------------------------------------------------
``structured_to_unstructured`` now returns a view, if the stride between the
fields is constant. Prior, padding between the fields or a reversed field
would lead to a copy.
This change only applies to ``ndarray``, ``memmap`` and ``recarray``. For all
other array subclasses, the behavior remains unchanged.

(`gh-23652 <https://github.com/numpy/numpy/pull/23652>`__)

Signed and unsigned integers always compare correctly
-----------------------------------------------------
When ``uint64`` and ``int64`` are mixed in NumPy, NumPy typically
promotes both to ``float64``.  This behavior may be argued about
but is confusing for comparisons ``==``, ``<=``, since the results
returned can be incorrect but the conversion is hidden since the
result is a boolean.
NumPy will now return the correct results for these by avoiding
the cast to float.

(`gh-23713 <https://github.com/numpy/numpy/pull/23713>`__)


Performance improvements and changes
====================================

Faster ``np.argsort`` on AVX-512 enabled processors
---------------------------------------------------
32-bit and 64-bit quicksort algorithm for np.argsort gain up to 6x speed up on
processors that support AVX-512 instruction set.

Thanks to `Intel corporation <https://open.intel.com/>`_ for sponsoring this
work.

(`gh-23707 <https://github.com/numpy/numpy/pull/23707>`__)

Faster ``np.sort`` on AVX-512 enabled processors
------------------------------------------------
Quicksort for 16-bit and 64-bit dtypes gain up to 15x and 9x speed up on
processors that support AVX-512 instruction set.

Thanks to `Intel corporation <https://open.intel.com/>`_ for sponsoring this
work.

(`gh-22315 <https://github.com/numpy/numpy/pull/22315>`__)

``__array_function__`` machinery is now much faster
---------------------------------------------------
The overhead of the majority of functions in NumPy is now smaller
especially when keyword arguments are used.  This change significantly
speeds up many simple function calls.

(`gh-23020 <https://github.com/numpy/numpy/pull/23020>`__)

``ufunc.at`` can be much faster
-------------------------------
Generic ``ufunc.at`` can be up to 9x faster. The conditions for this speedup:

- operands are aligned
- no casting

If ufuncs with appropriate indexed loops on 1d arguments with the above
conditions, ``ufunc.at`` can be up to 60x faster (an additional 7x speedup).
Appropriate indexed loops have been added to ``add``, ``subtract``,
``multiply``, ``floor_divide``, ``maximum``, ``minimum``, ``fmax``, and
``fmin``.

The internal logic is similar to the logic used for regular ufuncs, which also
have fast paths.

Thanks to the `D. E. Shaw group <https://deshaw.com/>`_ for sponsoring this
work.

(`gh-23136 <https://github.com/numpy/numpy/pull/23136>`__)

Faster membership test on ``NpzFile``
-------------------------------------
Membership test on ``NpzFile`` will no longer
decompress the archive if it is successful.

(`gh-23661 <https://github.com/numpy/numpy/pull/23661>`__)


Changes
=======

``np.r_[]`` and ``np.c_[]`` with certain scalar values
------------------------------------------------------
In rare cases, using mainly ``np.r_`` with scalars can lead to different
results.  The main potential changes are highlighted by the following::

    >>> np.r_[np.arange(5, dtype=np.uint8), -1].dtype
    int16  # rather than the default integer (int64 or int32)
    >>> np.r_[np.arange(5, dtype=np.int8), 255]
    array([  0,   1,   2,   3,   4, 255], dtype=int16)

Where the second example returned::

    array([ 0,  1,  2,  3,  4, -1], dtype=int8)

The first one is due to a signed integer scalar with an unsigned integer
array, while the second is due to ``255`` not fitting into ``int8`` and
NumPy currently inspecting values to make this work.
(Note that the second example is expected to change in the future due to
:ref:`NEP 50 <NEP50>`; it will then raise an error.)

(`gh-22539 <https://github.com/numpy/numpy/pull/22539>`__)

Most NumPy functions are wrapped into a C-callable
--------------------------------------------------
To speed up the ``__array_function__`` dispatching, most NumPy functions
are now wrapped into C-callables and are not proper Python functions or
C methods.
They still look and feel the same as before (like a Python function), and this
should only improve performance and user experience (cleaner tracebacks).
However, please inform the NumPy developers if this change confuses your
program for some reason.

(`gh-23020 <https://github.com/numpy/numpy/pull/23020>`__)

C++ standard library usage
--------------------------
NumPy builds now depend on the C++ standard library, because
the ``numpy.core._multiarray_umath`` extension is linked with
the C++ linker.

(`gh-23601 <https://github.com/numpy/numpy/pull/23601>`__)
