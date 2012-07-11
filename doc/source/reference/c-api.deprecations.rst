C API Deprecations
==================

Background
----------

The API exposed by NumPy for third-party extensions has grown over
years of releases, and has allowed programmers to directly access
NumPy functionality from C. This API can be best described as
"organic".  The core API originated with Numeric in 1995 and there are
patterns such as the heavy use of macros written to mimic Python's
C-API as well as account for compiler technology of the late 90's.
There is only a small group of volunteers who have had very little
time to spend on improving this API.   

There is an ongoing effort to improve the API to do things like remove
a "fortran" macro and ensure that the NPY_ prefixes don't collide with
names from the PyArray_ prefixes.  It is important in this effort,
however, to ensure that code that compiles for NumPy 1.X continues to
compile for NumPy 1.X.  At the same, time certain API's will be marked
as deprecated so that future-looking code can avoid these API's and
follow better practices. 

Another important role played by deprecations in the C API is to move
towards hiding internal details of the NumPy implementation. For those
needing direct, easy, access to the data of ndarrays, this will not
remove this ability. Rather, there are many potential performance
optimizations which require changing the implementation details, and
NumPy developers have been unable to try them because of the high
value of preserving ABI compatibility. By deprecating this direct
access, we will in the future be able to improve NumPy's performance
in ways we cannot presently.

Deprecation Mechanism NPY_NO_DEPRECATED_API
-------------------------------------------

In C, there is no equivalent to the deprecation warnings that Python
supports. One way to do deprecations is to flag them in the documentation
and release notes, then remove or change the deprecated features in a
future version (NumPy 2.0 and beyond).  

To use the NPY_NO_DEPRECATED_API mechanism, you need to #define it to
the target API version of NumPy before #including any NumPy headers.
If you want to confirm that your code is clean against 1.7, use::

    #define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

On compilers which support a #warning mechanism, NumPy issues a
compiler warning if you do not define the symbol NPY_NO_DEPRECATED_API.
This way, the fact that there are deprecations will be flagged for
third-party developers who may not have read the release notes closely.
