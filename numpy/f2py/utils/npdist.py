"""
These are pretty much deprecated, but should have never been in __init__.py
"""

import os
import sys
import subprocess

def compile(source,
            modulename='untitled',
            extra_args='',
            verbose=True,
            source_fn=None,
            extension='.f',
            full_output=False
           ):
    """
    Build extension module from a Fortran 77 source string with f2py.

    Parameters
    ----------
    source : str or bytes
        Fortran source of module / subroutine to compile

        .. versionchanged:: 1.16.0
           Accept str as well as bytes

    modulename : str, optional
        The name of the compiled python module
    extra_args : str or list, optional
        Additional parameters passed to f2py

        .. versionchanged:: 1.16.0
            A list of args may also be provided.

    verbose : bool, optional
        Print f2py output to screen
    source_fn : str, optional
        Name of the file where the fortran source is written.
        The default is to use a temporary file with the extension
        provided by the `extension` parameter
    extension : {'.f', '.f90'}, optional
        Filename extension if `source_fn` is not provided.
        The extension tells which fortran standard is used.
        The default is `.f`, which implies F77 standard.

        .. versionadded:: 1.11.0

    full_output : bool, optional
        If True, return a `subprocess.CompletedProcess` containing
        the stdout and stderr of the compile process, instead of just
        the status code.

        .. versionadded:: 1.20.0


    Returns
    -------
    result : int or `subprocess.CompletedProcess`
        0 on success, or a `subprocess.CompletedProcess` if
        ``full_output=True``

    Examples
    --------
    .. literalinclude:: ../../source/f2py/code/results/compile_session.dat
        :language: python

    """
    import tempfile
    import shlex

    if source_fn is None:
        f, fname = tempfile.mkstemp(suffix=extension)
        # f is a file descriptor so need to close it
        # carefully -- not with .close() directly
        os.close(f)
    else:
        fname = source_fn

    if not isinstance(source, str):
        source = str(source, 'utf-8')
    try:
        with open(fname, 'w') as f:
            f.write(source)

        args = ['-c', '-m', modulename, f.name]

        if isinstance(extra_args, str):
            is_posix = (os.name == 'posix')
            extra_args = shlex.split(extra_args, posix=is_posix)

        args.extend(extra_args)

        c = [sys.executable,
             '-c',
             'import numpy.f2py; numpy.f2py.main()'] + args
        try:
            cp = subprocess.run(c, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE)
        except OSError:
            # preserve historic status code used by exec_command()
            cp = subprocess.CompletedProcess(c, 127, stdout=b'', stderr=b'')
        else:
            if verbose:
                print(cp.stdout.decode())
    finally:
        if source_fn is None:
            os.remove(fname)

    if full_output:
        return cp
    else:
        return cp.returncode
