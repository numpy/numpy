*************************
NumPy 2.0 migration guide
*************************

This document contains a set of instructions on how to update your code to
work with Numpy 2.0.


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
a similar language may require updating to accomodate user input if they
are using the ``long`` or equivalent type on the C-side.
In this case, you may wish to use ``intp`` and cast user input or support
both ``long`` and ``intp`` (to better support NumPy 1.x as well).
When creating a new integer array in C or Cython, the new ``NPY_DEFAULT_INT``
macro will evaluate to either ``NPY_LONG`` or ``NPY_INTP`` depending on the
NumPy version.

Note that the NumPy random API is not affected by this change.

C-API Changes
=============
Some definitions where removed or replaced due to being outdated or
unmaintaibale.  Some new API definition will evaluate differently at
runtime between NumPy 2.0 and NumPy 1.x.
Some are defined in ``numpy/_core/include/numpy/npy_2_compat.h``
(for example ``NPY_DEFAULT_INT``) which can be vendored in full or part
to have the definitions available when compiling against NumPy 1.x.

If necessary, ``PyArray_RUNTIME_VERSION >= NPY_2_0_API_VERSION`` can be
used to explicitly implement different behavior on NumPy 1.x and 2.0.
(The compat header defines it in a way compatible with such use.)

Please let us know if you require additional workarounds here.

.. _migration_maxdims:

Increased maximum number of dimensions
--------------------------------------
The maximum number of dimensions (and arguments) was increased to 64, this
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


Namespace changes
=================

In NumPy 2.0 certain functions, modules, and constants were moved or removed
to make the NumPy namespace more userfriendly by removing unnecessary or
outdated functionality and clarifying which parts of NumPy are considered
private.
Please see the tables below for guidance on migration.  For most changes this
means replacing it with a backwards compatible alternative. 

Please refer to `NEP 52 <https://numpy.org/neps/nep-0052-python-api-cleanup.html>`_ for more details.

Main namespace
--------------

About 100 members of the main ``np`` namespace has been deprecated, removed, or
moved to a new place. It was done to reduce clutter and establish only one way to
access a given attribute. The table below shows members that have been removed:

======================  =================================================================
removed member          migration guideline
======================  =================================================================
add_docstring           It's still available as ``np.lib.add_docstring``.
add_newdoc              It's still available as ``np.lib.add_newdoc``.
add_newdoc_ufunc        It's an internal function and doesn't have a replacement.
asfarray                Use ``np.asarray`` with a float dtype instead.
byte_bounds             Now it's available under ``np.lib.array_utils.byte_bounds``
cast                    Use ``np.asarray(arr, dtype=dtype)`` instead.
cfloat                  Use ``np.complex128`` instead.
clongfloat              Use ``np.clongdouble`` instead.
compat                  There's no replacement, as Python 2 is no longer supported.
complex\_               Use ``np.complex128`` instead.
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
row_stack         Use ``np.vstack`` instead (``row_stack`` was an alias for ``v_stack``).
trapz             Use ``scipy.integrate.trapezoid`` instead.
================= =======================================================================


Finally, a set of internal enums has been removed. As they weren't used in
downstream libraries we don't provide any information on how to replace them:

[``FLOATING_POINT_SUPPORT``, ``FPE_DIVIDEBYZERO``, ``FPE_INVALID``, ``FPE_OVERFLOW``, 
``FPE_UNDERFLOW``, ``UFUNC_BUFSIZE_DEFAULT``, ``UFUNC_PYVALS_NAME``, ``CLIP``, ``WRAP``, 
``RAISE``, ``BUFSIZE``, ``ALLOW_THREADS``, ``MAXDIMS``, ``MAY_SHARE_EXACT``, 
``MAY_SHARE_BOUNDS``]


Lib namespace
-------------

Most of the functions available within ``np.lib`` are also present in the main
namespace, which is their primary location. To make it unambiguous how to access each
public function, ``np.lib`` is now empty and contains only a handful of specialized submodules,
classes and functions:

- ``array_utils``, ``format``, ``introspect``, ``mixins``, ``npyio``
  and ``stride_tricks`` submodules,

- ``Arrayterator`` and ``NumpyVersion`` classes,

- ``add_docstring`` and ``add_newdoc`` functions,

- ``tracemalloc_domain`` constant.

If you get an ``AttributeError`` when accessing an attribute from ``np.lib`` you should
try accessing it from the main ``np`` namespace then. If an item is also missing from
the main namespace, then you're using a private member. You should either use the existing
API or, in case it's infeasible, reach out to us with a request to restore the removed entry.


Core namespace
--------------

``np.core`` namespace is now officially private and has been renamed to ``np._core``.
The user should never fetch members from the ``_core`` directly - instead the main 
namespace should be used to access the attribute in question. The layout of the ``_core``
module might change in the future without notice, contrary to public modules which adhere 
to the deprecation period policy. If an item is also missing from the main namespace,
then you should either use the existing API or, in case it's infeasible, reach out to us
with a request to restore the removed entry.


ndarray and scalar namespace
----------------------------

A few methods from ``np.ndarray`` and ``np.generic`` scalar classes have been removed.
The table below provides replacements for the removed members:

======================  ========================================================
expired member          migration guideline
======================  ========================================================
newbyteorder            Use ``arr.view(arr.dtype.newbyteorder(order))`` instead.
ptp                     Use ``np.ptp(arr, ...)`` instead.
setitem                 Use ``arr[index] = value`` instead.
...                     ...
======================  ========================================================


Note about pickled files
------------------------

NumPy 2.0 is designed to load pickle files created with NumPy 1.26,
and vice versa. For versions 1.25 and earlier loading NumPy 2.0
pickle file will throw an exception.
