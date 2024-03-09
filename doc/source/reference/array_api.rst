.. _array-api-standard-compatibility:

********************************
Array API standard compatibility
********************************

NumPy's main namespace as well as the `numpy.fft` and `numpy.linalg` namespaces
are compatible [1]_ with the
`2022.12 version <https://data-apis.org/array-api/2022.12/index.html>`__
of the Python array API standard.

NumPy aims to implement support for the
`2023.12 version <https://data-apis.org/array-api/2023.12/index.html>`__
and future versions of the standard - assuming that those future versions can be
upgraded to given NumPy's
`backwards compatibility policy <https://numpy.org/neps/nep-0023-backwards-compatibility.html>`__.

For usage guidelines for downstream libraries and end users who want to write
code that will work with both NumPy and other array libraries, we refer to the
documentation of the array API standard itself and to code and
developer-focused documention in SciPy and scikit-learn.

Note that in order to use standard-complaint code with older NumPy versions
(< 2.0), the `array-api-compat
<https://github.com/data-apis/array-api-compat>`__ package may be useful. For
testing whether NumPy-using code is only using standard-compliant features
rather than anything NumPy-specific, the `array-api-strict
<https://github.com/data-apis/array-api-strict>`__ package can be used.

.. admonition:: History

    NumPy 1.22.0 was the first version to include support for the array API
    standard, via a separate ``numpy.array_api`` submodule. This module was
    marked as experimental (it emitted a warning on import) and removed in
    NumPy 2.0 because full support was included in the main namespace.
    `NEP 47 <https://numpy.org/neps/nep-0047-array-api-standard.html>`__ and
    `NEP 56 <https://numpy.org/neps/nep-0056-array-api-main-namespace.html>`__
    describe the motivation and scope for implementing the array API standard
    in NumPy.


Entry point
===========

NumPy installs an `entry point <https://packaging.python.org/en/latest/specifications/entry-points/>`__
that can be used for discovery purposes::

    >>> from importlib.metadata import entry_points
    >>> entry_points(group='array_api', name='numpy')
    [EntryPoint(name='numpy', value='numpy', group='array_api')]

Note that leaving out ``name='numpy'`` will cause a list of entry points to be
returned for all array API standard compatible implementations that installed
an entry point.


.. rubric:: Footnotes

.. [1] With a few very minor exceptions, as documented in
   `NEP 56 <https://numpy.org/neps/nep-0056-array-api-main-namespace.html>`__.
   The ``sum``, ``prod`` and ``trace`` behavior adheres to the 2023.12 version
   instead, as do function signatures; the only known incompatibility that may
   remain is that the standard forbids unsafe casts for in-place operators
   while NumPy supports those.
