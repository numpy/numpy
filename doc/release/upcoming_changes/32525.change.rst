`numpy.ma.cov` and `numpy.ma.corrcoef` use pairwise-complete observations
-------------------------------------------------------------------------
For masked arrays, each entry of `numpy.ma.cov` and `numpy.ma.corrcoef` is now
computed from the observations where both variables are unmasked, with the
means and variances taken over that same set.  Previously each variable was
centered on the mean over its own unmasked observations, while the sum of
products and the divisor ran over the shared ones, and `numpy.ma.corrcoef`
took its variances from the diagonal of `numpy.ma.cov`.  Coefficients outside
``[-1, 1]`` were possible and no longer are.

`numpy.ma.corrcoef` also returns ``1.0`` for a single variable with at least
two observations and a non-zero variance, which is what `numpy.corrcoef`
returns and what the diagonal of the matrix for two or more variables already
held.  It returned ``numpy.ma.masked`` since 2.1.0, and ``1`` before that
regardless of whether any observations were present; a single variable with
fewer than two observations, or no variance, remains masked.

Two further changes follow.  `numpy.ma.corrcoef` clips its result to
``[-1, 1]``, as `numpy.corrcoef` does, and masks a pair whose shared
observations leave either variable constant, where it previously returned an
unmasked value.  The 2.1.0 release note for ``ma.corrcoef``
(gh-26285) is superseded: the pairwise standard deviations it removed are
restored, now matching a ``ma.cov`` that is itself pairwise.
