.. _numpy-2-migration-guide:

*************************
NumPy 2.0 migration guide
*************************

This document contains a set of instructions on how to update your code to
work with NumPy 2.0. It covers changes in NumPy's Python and C APIs.

.. note::

   Note that NumPy 2.0 also breaks binary compatibility - if you are
   distributing binaries for a Python package that depends on NumPy's C API,
   please see :ref:`numpy-2-abi-handling`.



Ruff plugin
===========

Many of the changes covered in the 2.0 release notes and in this migration
guide can be automatically adapted in downstream code with a dedicated
`Ruff <https://docs.astral.sh/ruff/>`__ rule, namely rule
`NPY201 <https://docs.astral.sh/ruff/rules/numpy2-deprecation/>`__.

You should install ``ruff>=0.4.8`` and add the ``NPY201`` rule to your
``pyproject.toml``::

    [tool.ruff.lint]
    select = ["NPY201"]

You can also apply the NumPy 2.0 rule directly from the command line::

    $ ruff check path/to/code/ --select NPY201


.. _migration_promotion_changes:

Changes to NumPy data type promotion
=====================================

NumPy 2.0 changes promotion (the result of combining dissimilar data types)
as per :ref:`NEP 50 <NEP50>`. Please see the NEP for details on this change.
It includes a table of example changes and a backwards compatibility section.

The largest backwards compatibility change is that the precision of scalars
is now preserved consistently.  Two examples are:

* ``np.float32(3) + 3.`` now returns a float32 when it previously returned
  a float64.
* ``np.array([3], dtype=np.float32) + np.float64(3)`` will now return a float64
  array.  (The higher precision of the scalar is not ignored.)

For floating point values, this can lead to lower precision results when
working with scalars.  For integers, errors or overflows are possible.

To solve this, you may cast explicitly.  Very often, it may also be a good
solution to ensure you are working with Python scalars via ``int()``,
``float()``, or ``numpy_scalar.item()``.

To track down changes, you can enable emitting warnings for changed behavior
(use ``warnings.simplefilter`` to raise it as an error for a traceback)::

  np._set_promotion_state("weak_and_warn")

which is useful during testing. Unfortunately,
running this may flag many changes that are irrelevant in practice.

.. _migration_windows_int64:

Windows default integer
=======================
The default integer used by NumPy is now 64bit on all 64bit systems (and
32bit on 32bit system).  For historic reasons related to Python 2 it was
previously equivalent to the C ``long`` type.
The default integer is now equivalent to ``np.intp``.

Most end-users should not be affected by this change.  Some operations will
use more memory, but some operations may actually become faster.
If you experience issues due to calling a library written in a compiled
language it may help to explicitly cast to a ``long``, for example with:
``arr = arr.astype("long", copy=False)``.

Libraries interfacing with compiled code that are written in C, Cython, or
a similar language may require updating to accommodate user input if they
are using the ``long`` or equivalent type on the C-side.
In this case, you may wish to use ``intp`` and cast user input or support
both ``long`` and ``intp`` (to better support NumPy 1.x as well).
When creating a new integer array in C or Cython, the new ``NPY_DEFAULT_INT``
macro will evaluate to either ``NPY_LONG`` or ``NPY_INTP`` depending on the
NumPy version.

Note that the NumPy random API is not affected by this change.

C-API Changes
=============

Some definitions were removed or replaced due to being outdated or
unmaintainable.  Some new API definitions will evaluate differently at
runtime between NumPy 2.0 and NumPy 1.x.
Some are defined in ``numpy/_core/include/numpy/npy_2_compat.h``
(for example ``NPY_DEFAULT_INT``) which can be vendored in full or part
to have the definitions available when compiling against NumPy 1.x.

If necessary, ``PyArray_RUNTIME_VERSION >= NPY_2_0_API_VERSION`` can be
used to explicitly implement different behavior on NumPy 1.x and 2.0.
(The compat header defines it in a way compatible with such use.)

Please let us know if you require additional workarounds here.

.. _migration_c_descr:

The ``PyArray_Descr`` struct has been changed
---------------------------------------------
One of the most impactful C-API changes is that the ``PyArray_Descr`` struct
is now more opaque to allow us to add additional flags and have
itemsizes not limited by the size of ``int`` as well as allow improving
structured dtypes in the future and not burden new dtypes with their fields.

Code which only uses the type number and other initial fields is unaffected.
Most code will hopefully mainly access the ``->elsize`` field, when the
dtype/descriptor itself is attached to an array (e.g. ``arr->descr->elsize``)
this is best replaced with ``PyArray_ITEMSIZE(arr)``.

Where not possible, new accessor functions are required:

* ``PyDataType_ELSIZE`` and ``PyDataType_SET_ELSIZE`` (note that the result
  is now ``npy_intp`` and not ``int``).
* ``PyDataType_ALIGNMENT``
* ``PyDataType_FIELDS``, ``PyDataType_NAMES``, ``PyDataType_SUBARRAY``
* ``PyDataType_C_METADATA``

Cython code should use Cython 3, in which case the change is transparent.
(Struct access is available for elsize and alignment when compiling only for
NumPy 2.)

For compiling with both 1.x and 2.x if you use these new accessors it is
unfortunately necessary to either define them locally via a macro like::

  #if NPY_ABI_VERSION < 0x02000000
    #define PyDataType_ELSIZE(descr) ((descr)->elsize)
  #endif

or adding ``npy2_compat.h`` into your code base and explicitly include it
when compiling with NumPy 1.x (as they are new API).
Including the file has no effect on NumPy 2.

Please do not hesitate to open a NumPy issue, if you require assistance or
the provided functions are not sufficient.

**Custom User DTypes:**
Existing user dtypes must now use :c:type:`PyArray_DescrProto` to define
their dtype and slightly modify the code. See note in :c:func:`PyArray_RegisterDataType`.

Functionality moved to headers requiring ``import_array()``
-----------------------------------------------------------
If you previously included only ``ndarraytypes.h`` you may find that some
functionality is not available anymore and requires the inclusion of
``ndarrayobject.h`` or similar.
This include is also needed when vendoring ``npy_2_compat.h`` into your own
codebase to allow use of the new definitions when compiling with NumPy 1.x.

Functionality which previously did not require import includes:

* Functions to access dtype flags: ``PyDataType_FLAGCHK``,
  ``PyDataType_REFCHK``, and the related ``NPY_BEGIN_THREADS_DESCR``.
* ``PyArray_GETITEM`` and ``PyArray_SETITEM``.

.. warning::
  It is important that the ``import_array()`` mechanism is used to ensure
  that the full NumPy API is accessible when using the ``npy_2_compat.h``
  header.  In most cases your extension module probably already calls it.
  However, if not we have added ``PyArray_ImportNumPyAPI()`` as a preferable
  way to ensure the NumPy API is imported.  This function is light-weight 
  when called multiple times so that you may insert it wherever it may be
  needed (if you wish to avoid setting it up at module import).

.. _migration_maxdims:

Increased maximum number of dimensions
--------------------------------------
The maximum number of dimensions (and arguments) was increased to 64. This
affects the ``NPY_MAXDIMS`` and ``NPY_MAXARGS`` macros.
It may be good to review their use, and we generally encourage you to
not use these macros (especially ``NPY_MAXARGS``), so that a future version of
NumPy can remove this limitation on the number of dimensions.

``NPY_MAXDIMS`` was also used to signal ``axis=None`` in the C-API, including
the ``PyArray_AxisConverter``.
The latter will return ``-2147483648`` as an axis (the smallest integer value).
Other functions may error with
``AxisError: axis 64 is out of bounds for array of dimension`` in which
case you need to pass ``NPY_RAVEL_AXIS`` instead of ``NPY_MAXDIMS``.
``NPY_RAVEL_AXIS`` is defined in the ``npy_2_compat.h`` header and runtime
dependent (mapping to 32 on NumPy 1.x and ``-2147483648`` on NumPy 2.x).

Complex types - Underlying type changes
---------------------------------------

The underlying C types for all of the complex types have been changed to use
native C99 types. While the memory layout of those types remains identical
to the types used in NumPy 1.x, the API is slightly different, since direct
field access (like ``c.real`` or ``c.imag``) is no longer possible.

It is recommended to use the functions ``npy_creal`` and ``npy_cimag``
(and the corresponding float and long double variants) to retrieve
the real or imaginary part of a complex number, as these will work with both
NumPy 1.x and with NumPy 2.x. New functions ``npy_csetreal`` and
``npy_csetimag``, along with compatibility macros ``NPY_CSETREAL`` and
``NPY_CSETIMAG`` (and the corresponding float and long double variants),
have been added for setting the real or imaginary part.

The underlying type remains a struct under C++ (all of the above still remains
valid).

This has implications for Cython. It is recommended to always use the native
typedefs ``cfloat_t``, ``cdouble_t``, ``clongdouble_t`` rather than the NumPy
types ``npy_cfloat``, etc, unless you have to interface with C code written
using the NumPy types. You can still write cython code using the ``c.real`` and
``c.imag`` attributes (using the native typedefs), but you can no longer use
in-place operators ``c.imag += 1`` in Cython's c++ mode.

Because NumPy 2 now includes ``complex.h`` code that uses a variable named
``I`` may see an error such as

.. code-block::C
   error: expected ‘)’ before ‘__extension__’
                    double I,

to use the name ``I`` requires an ``#undef I`` now.

.. note::
  NumPy 2.0.1 briefly included the ``#undef I`` to help users not already
  including ``complex.h``.


Changes to namespaces
=====================

In NumPy 2.0 certain functions, modules, and constants were moved or removed
to make the NumPy namespace more user-friendly by removing unnecessary or
outdated functionality and clarifying which parts of NumPy are considered
private.
Please see the tables below for guidance on migration.  For most changes this
means replacing it with a backwards compatible alternative. 

Please refer to :ref:`NEP52` for more details.

Main namespace
--------------

About 100 members of the main ``np`` namespace have been deprecated, removed, or
moved to a new place. It was done to reduce clutter and establish only one way to
access a given attribute. The table below shows members that have been removed:

======================  =================================================================
removed member          migration guideline
======================  =================================================================
add_docstring           It's still available as ``np.lib.add_docstring``.
add_newdoc              It's still available as ``np.lib.add_newdoc``.
add_newdoc_ufunc        It's an internal function and doesn't have a replacement.
alltrue                 Use ``np.all`` instead.
asfarray                Use ``np.asarray`` with a float dtype instead.
byte_bounds             Now it's available under ``np.lib.array_utils.byte_bounds``
cast                    Use ``np.asarray(arr, dtype=dtype)`` instead.
cfloat                  Use ``np.complex128`` instead.
charrarray              It's still available as ``np.char.chararray``.
clongfloat              Use ``np.clongdouble`` instead.
compare_chararrays      It's still available as ``np.char.compare_chararrays``.
compat                  There's no replacement, as Python 2 is no longer supported.
complex\_               Use ``np.complex128`` instead.
cumproduct              Use ``np.cumprod`` instead.
DataSource              It's still available as ``np.lib.npyio.DataSource``.
deprecate               Emit ``DeprecationWarning`` with ``warnings.warn`` directly,
                        or use ``typing.deprecated``.
deprecate_with_doc      Emit ``DeprecationWarning`` with ``warnings.warn`` directly,
                        or use ``typing.deprecated``.
disp                    Use your own printing function instead.
fastCopyAndTranspose    Use ``arr.T.copy()`` instead.
find_common_type        Use ``numpy.promote_types`` or ``numpy.result_type`` instead. 
                        To achieve semantics for the ``scalar_types`` argument, 
                        use ``numpy.result_type`` and pass the Python values ``0``, 
                        ``0.0``, or ``0j``.
format_parser           It's still available as ``np.rec.format_parser``.
get_array_wrap
float\_                 Use ``np.float64`` instead.
geterrobj               Use the np.errstate context manager instead.
Inf                     Use ``np.inf`` instead.
Infinity                Use ``np.inf`` instead.
infty                   Use ``np.inf`` instead.
issctype                Use ``issubclass(rep, np.generic)`` instead.
issubclass\_            Use ``issubclass`` builtin instead.
issubsctype             Use ``np.issubdtype`` instead.
mat                     Use ``np.asmatrix`` instead.
maximum_sctype          Use a specific dtype instead. You should avoid relying
                        on any implicit mechanism and select the largest dtype of
                        a kind explicitly in the code.
NaN                     Use ``np.nan`` instead.
nbytes                  Use ``np.dtype(<dtype>).itemsize`` instead.
NINF                    Use ``-np.inf`` instead.
NZERO                   Use ``-0.0`` instead.
longcomplex             Use ``np.clongdouble`` instead.
longfloat               Use ``np.longdouble`` instead.
lookfor                 Search NumPy's documentation directly.
obj2sctype              Use ``np.dtype(obj).type`` instead.
PINF                    Use ``np.inf`` instead.
product                 Use ``np.prod`` instead.
PZERO                   Use ``0.0`` instead.
recfromcsv              Use ``np.genfromtxt`` with comma delimiter instead.
recfromtxt              Use ``np.genfromtxt`` instead.
round\_                 Use ``np.round`` instead.
safe_eval               Use ``ast.literal_eval`` instead.
sctype2char             Use ``np.dtype(obj).char`` instead.
sctypes                 Access dtypes explicitly instead.
seterrobj               Use the np.errstate context manager instead.
set_numeric_ops         For the general case, use ``PyUFunc_ReplaceLoopBySignature``. 
                        For ndarray subclasses, define the ``__array_ufunc__`` method 
                        and override the relevant ufunc.
set_string_function     Use ``np.set_printoptions`` instead with a formatter 
                        for custom printing of NumPy objects.
singlecomplex           Use ``np.complex64`` instead.
string\_                Use ``np.bytes_`` instead.
sometrue                Use ``np.any`` instead.
source                  Use ``inspect.getsource`` instead.
tracemalloc_domain      It's now available from ``np.lib``.
unicode\_               Use ``np.str_`` instead.
who                     Use an IDE variable explorer or ``locals()`` instead.
======================  =================================================================

If the table doesn't contain an item that you were using but was removed in ``2.0``,
then it means it was a private member. You should either use the existing API or,
in case it's infeasible, reach out to us with a request to restore the removed entry.

The next table presents deprecated members, which will be removed in a release after ``2.0``:

================= =======================================================================
deprecated member migration guideline
================= =======================================================================
in1d              Use ``np.isin`` instead.
row_stack         Use ``np.vstack`` instead (``row_stack`` was an alias for ``vstack``).
trapz             Use ``np.trapezoid`` or a ``scipy.integrate`` function instead.
================= =======================================================================


Finally, a set of internal enums has been removed. As they weren't used in
downstream libraries we don't provide any information on how to replace them:

[``FLOATING_POINT_SUPPORT``, ``FPE_DIVIDEBYZERO``, ``FPE_INVALID``, ``FPE_OVERFLOW``, 
``FPE_UNDERFLOW``, ``UFUNC_BUFSIZE_DEFAULT``, ``UFUNC_PYVALS_NAME``, ``CLIP``, ``WRAP``, 
``RAISE``, ``BUFSIZE``, ``ALLOW_THREADS``, ``MAXDIMS``, ``MAY_SHARE_EXACT``, 
``MAY_SHARE_BOUNDS``]


numpy.lib namespace
-------------------

Most of the functions available within ``np.lib`` are also present in the main
namespace, which is their primary location. To make it unambiguous how to access each
public function, ``np.lib`` is now empty and contains only a handful of specialized submodules,
classes and functions:

- ``array_utils``, ``format``, ``introspect``, ``mixins``, ``npyio``, ``scimath``
  and ``stride_tricks`` submodules,

- ``Arrayterator`` and ``NumpyVersion`` classes,

- ``add_docstring`` and ``add_newdoc`` functions,

- ``tracemalloc_domain`` constant.

If you get an ``AttributeError`` when accessing an attribute from ``np.lib`` you should
try accessing it from the main ``np`` namespace then. If an item is also missing from
the main namespace, then you're using a private member. You should either use the existing
API or, in case it's infeasible, reach out to us with a request to restore the removed entry.


numpy.core namespace
--------------------

The ``np.core`` namespace is now officially private and has been renamed to ``np._core``.
The user should never fetch members from the ``_core`` directly - instead the main 
namespace should be used to access the attribute in question. The layout of the ``_core``
module might change in the future without notice, contrary to public modules which adhere 
to the deprecation period policy. If an item is also missing from the main namespace,
then you should either use the existing API or, in case it's infeasible, reach out to us
with a request to restore the removed entry.


ndarray and scalar methods
--------------------------

A few methods from ``np.ndarray`` and ``np.generic`` scalar classes have been removed.
The table below provides replacements for the removed members:

======================  ========================================================
expired member          migration guideline
======================  ========================================================
newbyteorder            Use ``arr.view(arr.dtype.newbyteorder(order))`` instead.
ptp                     Use ``np.ptp(arr, ...)`` instead.
setitem                 Use ``arr[index] = value`` instead.
======================  ========================================================


numpy.strings namespace
-----------------------

A new `numpy.strings` namespace has been created, where most of the string
operations are implemented as ufuncs. The old `numpy.char` namespace still is
available, and, wherever possible, uses the new ufuncs for greater performance.
We recommend using the `~numpy.strings` functions going forward. The
`~numpy.char` namespace may be deprecated in the future.


Other changes
=============


Note about pickled files
------------------------

NumPy 2.0 is designed to load pickle files created with NumPy 1.26,
and vice versa. For versions 1.25 and earlier loading NumPy 2.0
pickle file will throw an exception.


Adapting to changes in the ``copy`` keyword
-------------------------------------------

The :ref:`copy keyword behavior changes <copy-keyword-changes-2.0>` in
`~numpy.asarray`, `~numpy.array` and `ndarray.__array__
<numpy.ndarray.__array__>` may require these changes:

* Code using ``np.array(..., copy=False)`` can in most cases be changed to
  ``np.asarray(...)``. Older code tended to use ``np.array`` like this because
  it had less overhead than the default ``np.asarray`` copy-if-needed
  behavior. This is no longer true, and ``np.asarray`` is the preferred function.
* For code that explicitly needs to pass ``None``/``False`` meaning "copy if
  needed" in a way that's compatible with NumPy 1.x and 2.x, see
  `scipy#20172 <https://github.com/scipy/scipy/pull/20172>`__ for an example
  of how to do so.
* For any ``__array__`` method on a non-NumPy array-like object, ``dtype=None``
  and ``copy=None`` keywords must be added to the signature - this will work with older
  NumPy versions as well (although older numpy versions will never pass in ``copy`` keyword).
  If the keywords are added to the ``__array__`` signature, then for:

  * ``copy=True`` and any ``dtype`` value always return a new copy,
  * ``copy=None`` create a copy if required (for example by ``dtype``),
  * ``copy=False`` a copy must never be made. If a copy is needed to return a numpy array
    or satisfy ``dtype``, then raise an exception (``ValueError``).

Writing numpy-version-dependent code
------------------------------------

It should be fairly rare to have to write code that explicitly branches on the
``numpy`` version - in most cases, code can be rewritten to be compatible with
1.x and 2.0 at the same time. However, if it is necessary, here is a suggested
code pattern to use, using `numpy.lib.NumpyVersion`::

    # example with AxisError, which is no longer available in
    # the main namespace in 2.0, and not available in the
    # `exceptions` namespace in <1.25.0 (example uses <2.0.0b1
    # for illustrative purposes):
    if np.lib.NumpyVersion(np.__version__) >= '2.0.0b1':
        from numpy.exceptions import AxisError
    else:
        from numpy import AxisError

This pattern will work correctly including with NumPy release candidates, which
is important during the 2.0.0 release period.
