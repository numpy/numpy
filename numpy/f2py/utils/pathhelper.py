"""
For exposing source paths
"""

from pathlib import Path

def get_include():
    """
    Return the directory that contains the fortranobject.c and .h files.

    .. note::

        This function is not needed when building an extension with
        `numpy.distutils` directly from ``.f`` and/or ``.pyf`` files
        in one go.

    Python extension modules built with f2py-generated code need to use
    ``fortranobject.c`` as a source file, and include the ``fortranobject.h``
    header. This function can be used to obtain the directory containing
    both of these files.

    Returns
    -------
    include_path : str
        Absolute path to the directory containing ``fortranobject.c`` and
        ``fortranobject.h``.

    Notes
    -----
    .. versionadded:: 1.22.0

    Unless the build system you are using has specific support for f2py,
    building a Python extension using a ``.pyf`` signature file is a two-step
    process. For a module ``mymod``:

        - Step 1: run ``python -m numpy.f2py mymod.pyf --quiet``. This
          generates ``_mymodmodule.c`` and (if needed)
          ``_fblas-f2pywrappers.f`` files next to ``mymod.pyf``.
        - Step 2: build your Python extension module. This requires the
          following source files:

              - ``_mymodmodule.c``
              - ``_mymod-f2pywrappers.f`` (if it was generated in step 1)
              - ``fortranobject.c``

    See Also
    --------
    numpy.get_include : function that returns the numpy include directory

    """
    return Path(__file__).parent.parent.resolve().joinpath('csrcs')
