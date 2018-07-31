================================
NEP 17 — Split Out Masked Arrays
================================

:Author: Stéfan van der Walt <stefanv@berkeley.edu>
:Status: Rejected
:Type: Standards Track
:Created: 2018-03-22
:Resolution: https://mail.python.org/pipermail/numpy-discussion/2018-May/078026.html

Abstract
--------

This NEP proposes removing MaskedArray functionality from NumPy, and
publishing it as a stand-alone package.

Detailed description
--------------------

MaskedArrays are a sub-class of the NumPy ``ndarray`` that adds
masking capabilities, i.e. the ability to ignore or hide certain array
values during computation.

While historically convenient to distribute this class inside of NumPy,
improved packaging has made it possible to distribute it separately
without difficulty.

Motivations for this move include:

 * Focus: the NumPy package should strive to only include the
   `ndarray` object, and the essential utilities needed to manipulate
   such arrays.
 * Complexity: the MaskedArray implementation is non-trivial, and imposes
   a significant maintenance burden.
 * Compatibility: MaskedArray objects, being subclasses [1]_ of `ndarrays`,
   often cause complications when being used with other packages.
   Fixing these issues is outside the scope of NumPy development.

This NEP proposes a deprecation pathway through which MaskedArrays
would still be accessible to users, but no longer as part of the core
package.

Implementation
--------------

Currently, a MaskedArray is created as follows::

  from numpy import ma
  ma.array([1, 2, 3], mask=[True, False, True])

This will return an array where the values 1 and 3 are masked (no
longer visible to operations such as `np.sum`).

We propose refactoring the `np.ma` subpackage into a new
pip-installable library called `maskedarray` [2]_, which would be used
in a similar fashion::

  import maskedarray as ma
  ma.array([1, 2, 3], mask=[True, False, True])

For two releases of NumPy, `maskedarray` would become a NumPy
dependency, and would expose MaskedArrays under the existing name,
`np.ma`.  If imported as `np.ma`, a `NumpyDeprecationWarning` will
be raised, describing the impending deprecation with instructions on
how to modify code to use `maskedarray`.

After two releases, `np.ma` will be removed entirely. In order to obtain
`np.ma`, a user will install it via `pip install` or via their package
manager. Subsequently, `importing maskedarray` on a version of NumPy that
includes it intgrally will raise an `ImportError`.

Documentation
`````````````

NumPy's internal documentation refers explicitly to MaskedArrays in
certain places, e.g. `ndarray.concatenate`:

> When one or more of the arrays to be concatenated is a MaskedArray,
> this function will return a MaskedArray object instead of an ndarray,
> but the input masks are *not* preserved. In cases where a MaskedArray
> is expected as input, use the ma.concatenate function from the masked
> array module instead.

Such documentation will be removed, since the expectation is that
users of `maskedarray` will use methods from that package to operate
on MaskedArrays.

Other appearances
~~~~~~~~~~~~~~~~~

Explicit MaskedArray support will be removed from:

- `numpygenfromtext`
- `numpy.libmerge_arrays`, `numpy.lib.stack_arrays`

Backward compatibility
----------------------

For two releases of NumPy, apart from a deprecation notice, there will
be no user visible changes.  Thereafter, `np.ma` will no longer be
available (instead, MaskedArrays will live in the `maskedarray`
package).

Note also that new PEPs on array-like objects may eventually provide
better support for MaskedArrays than is currently available.

Alternatives
------------

After a lively discussion on the mailing list:

- There is support (and active interest in) making a better *new* masked array
  class.
- The new class should be a consumer of the external NumPy API with no special
  status (unlike today where there are hacks across the codebase to support it)
- `MaskedArray` will stay where it is, at least until the new masked array
  class materializes and has been tried in the wild.

References and Footnotes
------------------------

.. [1] Subclassing ndarray,
       https://docs.scipy.org/doc/numpy/user/basics.subclassing.html
.. [2] PyPi: maskedarray, https://pypi.org/project/maskedarray/

Copyright
---------

This document has been placed in the public domain.
