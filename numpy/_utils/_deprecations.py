"""
Module used for marking functions deprecated.
"""

import functools
import sys
import textwrap
import warnings

__all__ = ["deprecate", "deprecate_with_doc"]


class _Deprecate:
    """
    Decorator class to deprecate old functions.

    Refer to `deprecate` for details.

    See Also
    --------
    deprecate

    """

    def __init__(self, old_name=None, new_name=None, message=None):
        self.old_name = old_name
        self.new_name = new_name
        self.message = message

    def __call__(self, func, *args, **kwargs):
        """
        Decorator call.  Refer to ``decorate``.

        """
        old_name = self.old_name
        new_name = self.new_name
        message = self.message

        if old_name is None:
            old_name = func.__name__
        if new_name is None:
            depdoc = "`%s` is deprecated!" % old_name
        else:
            depdoc = "`%s` is deprecated, use `%s` instead!" % \
                     (old_name, new_name)

        if message is not None:
            depdoc += "\n" + message

        @functools.wraps(func)
        def newfunc(*args, **kwds):
            warnings.warn(depdoc, DeprecationWarning, stacklevel=2)
            return func(*args, **kwds)

        newfunc.__name__ = old_name
        doc = func.__doc__
        if doc is None:
            doc = depdoc
        else:
            lines = doc.expandtabs().split('\n')
            indent = _get_indent(lines[1:])
            if lines[0].lstrip():
                # Indent the original first line to let inspect.cleandoc()
                # dedent the docstring despite the deprecation notice.
                doc = indent * ' ' + doc
            else:
                # Remove the same leading blank lines as cleandoc() would.
                skip = len(lines[0]) + 1
                for line in lines[1:]:
                    if len(line) > indent:
                        break
                    skip += len(line) + 1
                doc = doc[skip:]
            depdoc = textwrap.indent(depdoc, ' ' * indent)
            doc = '\n\n'.join([depdoc, doc])
        newfunc.__doc__ = doc

        return newfunc


def _get_indent(lines):
    """
    Determines the leading whitespace that could be removed from all the lines.
    """
    indent = sys.maxsize
    for line in lines:
        content = len(line.lstrip())
        if content:
            indent = min(indent, len(line) - content)
    if indent == sys.maxsize:
        indent = 0
    return indent


def deprecate(*args, **kwargs):
    """
    Issues a DeprecationWarning, adds warning to `old_name`'s
    docstring, rebinds ``old_name.__name__`` and returns the new
    function object.

    This function may also be used as a decorator.

    Parameters
    ----------
    func : function
        The function to be deprecated.
    old_name : str, optional
        The name of the function to be deprecated. Default is None, in
        which case the name of `func` is used.
    new_name : str, optional
        The new name for the function. Default is None, in which case the
        deprecation message is that `old_name` is deprecated. If given, the
        deprecation message is that `old_name` is deprecated and `new_name`
        should be used instead.
    message : str, optional
        Additional explanation of the deprecation.  Displayed in the
        docstring after the warning.

    Returns
    -------
    old_func : function
        The deprecated function.

    Examples
    --------
    Note that ``olduint`` returns a value after printing Deprecation
    Warning:

    >>> olduint = np.deprecate(np.uint)
    DeprecationWarning: `uint64` is deprecated! # may vary
    >>> olduint(6)
    6

    """
    # Deprecate may be run as a function or as a decorator
    # If run as a function, we initialise the decorator class
    # and execute its __call__ method.

    if args:
        fn = args[0]
        args = args[1:]

        return _Deprecate(*args, **kwargs)(fn)
    else:
        return _Deprecate(*args, **kwargs)


def deprecate_with_doc(msg):
    """
    Deprecates a function and includes the deprecation in its docstring.

    This function is used as a decorator. It returns an object that can be
    used to issue a DeprecationWarning, by passing the to-be decorated
    function as argument, this adds warning to the to-be decorated function's
    docstring and returns the new function object.

    See Also
    --------
    deprecate : Decorate a function such that it issues a `DeprecationWarning`

    Parameters
    ----------
    msg : str
        Additional explanation of the deprecation. Displayed in the
        docstring after the warning.

    Returns
    -------
    obj : object

    """
    return _Deprecate(message=msg)
