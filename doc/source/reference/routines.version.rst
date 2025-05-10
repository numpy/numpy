.. currentmodule:: numpy.version

.. _routines.version:

*******************
Version information
*******************

The ``numpy.version`` submodule includes several constants that expose more
detailed information about the exact version of the installed ``numpy``
package:

.. data:: version

    Version string for the installed package - matches ``numpy.__version__``.

.. data:: full_version

    Version string - the same as ``numpy.version.version``.

.. data:: short_version

    Version string without any local build identifiers.

    .. rubric:: Examples

    >>> np.__version__
    '2.1.0.dev0+git20240319.2ea7ce0'  # may vary
    >>> np.version.short_version
    '2.1.0.dev0'  # may vary

.. data:: git_revision

    String containing the git hash of the commit from which ``numpy`` was built.

.. data:: release

    ``True`` if this version is a ``numpy`` release, ``False`` if a dev version.
