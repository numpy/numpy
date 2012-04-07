C API Deprecations
==================

Background
----------

The API exposed by NumPy for third-party extensions has grown over
years of releases, and has allowed programmers to directly access
NumPy functionality from C. This API was not originally designed to
take into account best practices for C APIs, and was maintained by
a small group of people with very little time to expend on improving
it.

Starting with NumPy 1.7, we are attempting to make a concerted effort to
clean up the API. This includes fixing some grave sins, like removing
the macro *fortran* (which was defined to *fortran_*) as well as
clarifying some confusing choices caused by designing separate
subsystems without noticing how they interacted. For example,
NPY_DEFAULT was a flag controlling ndarray construction, while
PyArray_DEFAULT was the default dtype enumeration value.

Another important role played by deprecations in the C API is to move
towards hiding internal details of the NumPy implementation. For those
needing direct, easy, access to the data of ndarrays, this will not
remove this ability. Rather, there are many potential performance
optimizations which require changing the implementation details, and
NumPy developers have been unable to try them because of the high
value of preserving ABI compatibility. By deprecating this direct access
over the course of several releases, we will in the future be able to
improve NumPy's performance in ways we cannot presently.

Deprecation Mechanism NPY_NO_DEPRECATED_API
-------------------------------------------

In C, there is no equivalent to the deprecation warnings that Python
supports. One way to do deprecations is to flag them in the documentation
and release notes, then remove or change the deprecated features in the
next version. We intend to do this, but also have created a mechanism to help
third-party developers test that their code does not use any of the
deprecated features.

To use the NPY_NO_DEPRECATED_API mechanism, you need to #define it to
the target API version of NumPy before #including any NumPy headers.
If you want to confirm that your code is clean against 1.7, use::

    #define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

On compilers which support a #warning mechanism, NumPy issues a
compiler warning if you do not define the symbol NPY_NO_DEPRECATED_API.
This way, the fact that there are deprecations will be flagged for
third-party developers who may not have read the release notes closely.
