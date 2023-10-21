*************************
NumPy 2.0 Migration Guide
*************************

This document contains a set of instructions on how to update your code to work with
the Numpy 2.0 Python API. Most of the changes are trivial, and require the end user
to use a different name/module to access a given function/constant.

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
issctype
issubclass\_            Use ``issubclass`` builtin instead.
issubsctype             Use ``np.issubdtype`` instead.
mat                     Use ``np.asmatrix`` instead.
maximum_sctype
NaN                     Use ``np.nan`` instead.
nbytes                  Use ``np.dtype(<dtype>).itemsize`` instead.
NINF                    Use ``-np.inf`` instead.
NZERO                   Use ``-0.0`` instead.
longcomplex             Use ``np.clongdouble`` instead.
longfloat               Use ``np.longdouble`` instead.      
lookfor                 Search NumPy's documentation directly.
obj2sctype
PINF                    Use ``np.inf`` instead.
PZERO                   Use ``0.0`` instead.
recfromcsv              Use ``np.genfromtxt`` with comma delimiter instead.
recfromtxt              Use ``np.genfromtxt`` instead.
round\_                 Use ``np.round`` instead.
safe_eval               Use ``ast.literal_eval`` instead.
sctype2char
sctypes
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
trapz             Use ``scipy.interpolate.trapezoid`` instead.
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


NDArray and scalar namespace
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
