"""Attach docstrings to objects defined in C (see ``numpy/_core/_add_newdocs.py``)."""
import inspect
import types
import warnings

from numpy._core import overrides
from numpy._core._multiarray_umath import add_docstring

__all__ = ["add_newdoc"]


def _needs_add_docstring(obj):
    """
    Returns true if the only way to set the docstring of `obj` from python is
    via add_docstring.

    This function errs on the side of being overly conservative.
    """
    Py_TPFLAGS_IMMUTABLETYPE = 1 << 8

    if isinstance(obj, (types.FunctionType, types.MethodType, property)):
        return False

    if isinstance(obj, type):
        # ``__doc__`` is read-only exactly on immutable types: static
        # types, plus heap types that set ``Py_TPFLAGS_IMMUTABLETYPE``.
        return bool(obj.__flags__ & Py_TPFLAGS_IMMUTABLETYPE)

    return True


def _add_docstring(obj, doc, warn_on_python):
    doc = inspect.cleandoc(doc)

    if warn_on_python and not _needs_add_docstring(obj):
        warnings.warn(
            f"add_newdoc was used on a pure-python object {obj}. "
            "Prefer to attach it directly to the source.",
            UserWarning,
            stacklevel=3)

    # For types, try to assign ``__doc__`` directly (works for heap types).
    # When that succeeds, ``add_docstring`` only needs to populate
    # ``__text_signature__`` from any ``"\n--\n\n"`` stub.  Static types
    # (where ``__doc__`` is read-only) fall through unchanged.
    if isinstance(obj, type):
        head, sep, body = doc.partition("\n--\n\n")
        try:
            obj.__doc__ = body if sep else doc
        except Exception:
            pass  # just assume we should use add_docstring.
        else:
            if not sep:
                return
            doc = head + sep  # set only text-signature part

    try:
        add_docstring(obj, doc)
    except Exception:
        pass


def add_newdoc(place, obj, doc, warn_on_python=True):
    """
    Add documentation to an existing object, typically one defined in C

    The purpose is to allow easier editing of the docstrings without requiring
    a re-compile. This exists primarily for internal use within numpy itself.

    Parameters
    ----------
    place : str
        The absolute name of the module to import from
    obj : str | None
        The name of the object to add documentation to, typically a class or
        function name.
    doc : str | tuple[str, str] | list[tuple[str, str]]
        If a string, the documentation to apply to `obj`

        If a tuple, then the first element is interpreted as an attribute
        of `obj` and the second as the docstring to apply -
        ``(method, docstring)``

        If a list, then each element of the list should be a tuple of length
        two - ``[(method1, docstring1), (method2, docstring2), ...]``
    warn_on_python : bool
        If True, the default, emit `UserWarning` if this is used to attach
        documentation to a pure-python object.

    Notes
    -----
    This routine never raises an error if the docstring can't be written, but
    will raise an error if the object being documented does not exist.

    This routine cannot modify read-only docstrings, as appear
    in new-style classes or built-in functions. Because this
    routine never raises an error the caller must check manually
    that the docstrings were changed.

    Since this function grabs the ``char *`` from a c-level str object and puts
    it into the ``tp_doc`` slot of the type of `obj`, it violates a number of
    C-API best-practices, by:

    - modifying a `PyTypeObject` after calling `PyType_Ready`
    - calling `Py_INCREF` on the str and losing the reference, so the str
      will never be released

    If possible it should be avoided.
    """
    new = getattr(__import__(place, globals(), {}, [obj]), obj)
    if isinstance(doc, str):
        if "${ARRAY_FUNCTION_LIKE}" in doc:
            doc = overrides.get_array_function_like_doc(new, doc)
        _add_docstring(new, doc, warn_on_python)
    elif isinstance(doc, tuple):
        attr, docstring = doc
        _add_docstring(getattr(new, attr), docstring, warn_on_python)
    elif isinstance(doc, list):
        for attr, docstring in doc:
            _add_docstring(getattr(new, attr), docstring, warn_on_python)
