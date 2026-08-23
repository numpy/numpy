:orphan:

*************
Miscellaneous
*************

IEEE 754 floating point special values
--------------------------------------

Special values defined in numpy: :data:`~numpy.nan`, :data:`~numpy.inf`

NaNs can be used as a poor-man's mask (if you don't care what the
original value was)

Note: cannot use equality to test NaNs. E.g.: ::

 >>> myarr = np.array([1., 0., np.nan, 3.])
 >>> np.nonzero(myarr == np.nan)
 (array([], dtype=int64),)

::

 >>> np.nan == np.nan  # is always False! Use special numpy functions instead.
 False

::

 >>> myarr[myarr == np.nan] = 0. # doesn't work
 >>> myarr
 array([  1.,   0.,  nan,   3.])

::

 >>> myarr[np.isnan(myarr)] = 0. # use this instead find
 >>> myarr
 array([1.,  0.,  0.,  3.])

Other related special value functions:

- :func:`~numpy.isnan` - True if value is nan
- :func:`~numpy.isinf` - True if value is inf
- :func:`~numpy.isfinite` - True if not nan or inf
- :func:`~numpy.nan_to_num` - Map nan to 0, inf to max float, -inf to min float

The following corresponds to the usual functions except that nans are excluded
from the results:

- :func:`~numpy.nansum`
- :func:`~numpy.nanmax`
- :func:`~numpy.nanmin`
- :func:`~numpy.nanargmax`
- :func:`~numpy.nanargmin`

 >>> x = np.arange(10.)
 >>> x[3] = np.nan
 >>> x.sum()
 nan
 >>> np.nansum(x)
 42.0
