.. _how-to-numerical-issues:

How to diagnose numerical issues (precision, overflow, NaNs)
===========================================================

This how-to helps you quickly **identify and fix** common numerical issues in
NumPy: precision loss, overflow/underflow, and NaNs/Infs. It complements the
API reference for :func:`numpy.seterr`, :func:`numpy.errstate`,
:func:`numpy.isclose`, :func:`numpy.allclose`, :class:`numpy.finfo`,
and :func:`numpy.sum`.

TL;DR checklist
---------------

- **Start with dtypes**: know if your data are ``float32`` or ``float64``.
- **Time/inspect** a small repro before changing algorithms.
- **Use tolerances for equality**: :func:`numpy.isclose`, :func:`numpy.allclose`
  (set both ``rtol`` and ``atol``; near zero, ``atol`` matters).
- **Handle warnings explicitly** with :func:`numpy.errstate` /
  :func:`numpy.seterr` during sensitive operations.
- **Accumulate in higher precision** when appropriate, e.g.
  ``np.sum(x, dtype=np.float64)`` for ``float32`` inputs.
- **Know machine limits** with :class:`numpy.finfo` and spacing via
  :func:`numpy.spacing`.

Check dtypes and machine limits
-------------------------------

.. code-block:: python

   import numpy as np

   x = np.asarray(x)              # your data
   print(x.dtype)                 # float32 or float64?
   print(np.finfo(x.dtype))       # eps, min/max, tiny (underflow threshold)

See :class:`numpy.finfo` and :func:`numpy.spacing` for precision granularity.  These
values explain when rounding or underflow is expected.  (:class:`numpy.finfo`,
:func:`numpy.spacing`)

Compare with tolerances (don’t use ==)
--------------------------------------

Floating-point results that should be "equal" often differ by tiny amounts.
Prefer :func:`numpy.isclose` / :func:`numpy.allclose`:

.. code-block:: python

   a = np.array([1.0, 1e-12, 1e-9])
   b = a * (1 + 1e-12)
   near = np.isclose(a, b, rtol=1e-10, atol=1e-12)   # elementwise
   ok   = np.allclose(a, b, rtol=1e-10, atol=1e-12)  # single boolean

Notes:

- Near zero, ``atol`` dominates; increase it to avoid false negatives.
- Use ``equal_nan=True`` if NaNs in matching positions should compare equal.

Handle warnings and control FP behavior
---------------------------------------

Use :func:`numpy.errstate` as a context manager to **see or silence** warnings
precisely where they occur:

.. code-block:: python

   with np.errstate(over='warn', divide='raise', invalid='warn', under='ignore'):
       y = np.log(x)          # trigger warnings here only
       z = (x / 0.0)          # may raise if divide='raise'

See :func:`numpy.seterr`, :func:`numpy.geterr`, and the topic page
:ref:`floating point error handling <routines.err>`.  (Under-the-hood
defaults changed over time; consult the current docs.)

Diagnose precision loss in reductions (sum/mean)
------------------------------------------------

``np.sum`` and related reductions may accumulate rounding error. NumPy often
uses pairwise (tree) summation for better accuracy, but not in every case.

Quick mitigations:

- **Accumulate in higher precision**:

  .. code-block:: python

     s32 = np.sum(x32, dtype=np.float64)   # x32 is float32 array
     m32 = np.mean(x32, dtype=np.float64)

- **Order-insensitive check** for stability: shuffle the input and compare
  results with :func:`numpy.allclose`.

See Notes in :func:`numpy.sum` for accuracy details.

Avoid unstable formulas
-----------------------

- Subtracting nearly equal numbers magnifies error. Prefer **algebraically
  stable** alternatives (e.g., log-sum-exp for softmax):

  .. code-block:: python

     def logsumexp(v):
         vmax = v.max()
         return vmax + np.log(np.exp(v - vmax).sum())

- Normalize with safe denominators:

  .. code-block:: python

     denom = np.linalg.norm(x)
     y = x / denom if denom > 0 else x

Triage NaNs/Infs
----------------

.. code-block:: python

   bad = ~np.isfinite(x)       # True where x is NaN or +/-Inf
   if bad.any():
       # decide: drop, clip, or impute
       x = np.where(bad, 0.0, x)

Recipe: robust equality test
----------------------------

.. code-block:: python

   def close_enough(a, b, scale=1.0):
       # scale sets a problem-appropriate absolute tolerance
       return np.allclose(a, b, rtol=1e-10, atol=1e-12*scale, equal_nan=False)

See also
--------

- :func:`numpy.isclose`, :func:`numpy.allclose`
- :class:`numpy.finfo`, :func:`numpy.spacing`
- :func:`numpy.seterr`, :func:`numpy.errstate`, :ref:`routines.err <routines.err>`
- :func:`numpy.sum` (Notes on numerical accuracy)

Contributor notes
-----------------

- Keep snippets **short** and focused; prefer ``.. code-block:: python`` over
  doctests for longer fragments.
- Cross-link to reference pages rather than duplicating details.
