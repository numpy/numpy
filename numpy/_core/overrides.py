"""Implementation of __array_function__ overrides from NEP-18."""
import collections
import functools
import inspect

from numpy._core._multiarray_umath import (
    _ArrayFunctionDispatcher,
    _get_implementing_args,
    add_docstring,
)
from numpy._globals import _NoValue
from numpy._utils import set_module  # noqa: F401
from numpy._utils._inspect import getargspec

ARRAY_FUNCTIONS = set()

# ufunc.reduce positional argument order, used by the exact-ndarray
# reduction fast path.  Public reduction functions (np.sum, np.any, ...)
# use these same parameter names, which is how their signatures are
# mapped onto a direct ufunc.reduce call in C.
_REDUCE_SLOTS = {
    "a": 0, "axis": 1, "dtype": 2, "out": 3,
    "keepdims": 4, "initial": 5, "where": 6,
}
# Values for reduce arguments the caller did not pass (or passed as
# np._NoValue).  Slot 0 (`a`) has no default: it is required.
_REDUCE_DEFAULTS = (None, None, None, None, False, _NoValue, True)


def _resolve_reduction_spec(implementation, ufunc, defaults_override):
    """Map ``implementation``'s parameters onto ``ufunc.reduce`` arguments.

    Returns ``(reduce_callable, slots, defaults)`` where ``slots[i]`` is the
    reduce-argument index of the i-th parameter (aligned with the
    ``param_names`` table from ``_resolve_relevant_arg_spec``) and
    ``defaults`` holds the reduce-call value for every argument the caller
    omitted.  The C dispatcher uses this to call ``ufunc.reduce`` directly
    for exact-ndarray calls, skipping the Python wrapper entirely.
    """
    defaults = list(_REDUCE_DEFAULTS)
    if defaults_override:
        for name, value in defaults_override.items():
            defaults[_REDUCE_SLOTS[name]] = value
    slots = []
    for name, param in inspect.signature(implementation).parameters.items():
        if (name not in _REDUCE_SLOTS
                or param.kind not in (param.POSITIONAL_OR_KEYWORD,
                                      param.KEYWORD_ONLY)):
            raise RuntimeError(
                f"reduction fast path cannot map parameter {name!r} of "
                f"{implementation.__qualname__} onto ufunc.reduce")
        slots.append(_REDUCE_SLOTS[name])
    return (ufunc.reduce, tuple(slots), tuple(defaults))


array_function_like_doc = (
    """like : array_like, optional
        Reference object to allow the creation of arrays which are not
        NumPy arrays. If an array-like passed in as ``like`` supports
        the ``__array_function__`` protocol, the result will be defined
        by it. In this case, it ensures the creation of an array object
        compatible with that passed in via this argument."""
)

def get_array_function_like_doc(public_api, docstring_template=""):
    ARRAY_FUNCTIONS.add(public_api)
    docstring = public_api.__doc__ or docstring_template
    return docstring.replace("${ARRAY_FUNCTION_LIKE}", array_function_like_doc)

def finalize_array_function_like(public_api):
    public_api.__doc__ = get_array_function_like_doc(public_api)
    return public_api


add_docstring(
    _ArrayFunctionDispatcher,
    """
    Class to wrap functions with checks for __array_function__ overrides.

    The first two arguments are required and can only be passed by position.

    Parameters
    ----------
    dispatcher : function or None
        The dispatcher function that returns a single sequence-like object
        of all arguments relevant.  It must have the same signature (except
        the default values) as the actual implementation.
        If ``None``, this is a ``like=`` dispatcher and the
        ``_ArrayFunctionDispatcher`` must be called with ``like`` as the
        first (additional and positional) argument.
    implementation : function
        Function that implements the operation on NumPy arrays without
        overrides.  Arguments passed calling the ``_ArrayFunctionDispatcher``
        will be forwarded to this (and the ``dispatcher``) as if using
        ``*args, **kwargs``.
    reduction : tuple or None, optional
        Private internal configuration for the exact-ndarray reduction path.

    Attributes
    ----------
    _implementation : function
        The original implementation passed in.
    """)


# exposed for testing purposes; used internally by _ArrayFunctionDispatcher
add_docstring(
    _get_implementing_args,
    """
    Collect arguments on which to call __array_function__.

    Parameters
    ----------
    relevant_args : iterable of array-like
        Iterable of possibly array-like arguments to check for
        __array_function__ methods.

    Returns
    -------
    Sequence of arguments with __array_function__ methods, in the order in
    which they should be called.  Returns an empty sequence when every
    argument is an exact ``ndarray`` or a basic Python type (the caller
    short-circuits to the default implementation in that case).
    """)


ArgSpec = collections.namedtuple('ArgSpec', 'args varargs keywords defaults')


def verify_matching_signatures(implementation, dispatcher):
    """Verify that a dispatcher function has the right signature."""
    implementation_spec = ArgSpec(*getargspec(implementation))
    dispatcher_spec = ArgSpec(*getargspec(dispatcher))

    if (implementation_spec.args != dispatcher_spec.args or
            implementation_spec.varargs != dispatcher_spec.varargs or
            implementation_spec.keywords != dispatcher_spec.keywords or
            (bool(implementation_spec.defaults) !=
             bool(dispatcher_spec.defaults)) or
            (implementation_spec.defaults is not None and
             len(implementation_spec.defaults) !=
             len(dispatcher_spec.defaults))):
        raise RuntimeError(f'implementation and dispatcher for {implementation} have '
                           'different function signatures')

    if implementation_spec.defaults is not None:
        if dispatcher_spec.defaults != (None,) * len(dispatcher_spec.defaults):
            raise RuntimeError('dispatcher functions can only use None for '
                               'default argument values')


def _resolve_relevant_arg_spec(implementation, relevant_arg_names):
    """Resolve arg names into parameter indices against ``implementation``'s
    signature.  Rejects an empty spec, ``*args`` (positions would be
    ambiguous at call time), and unknown names.

    An index ``i`` encodes both lookup channels: the arg may be matched
    positionally iff ``i < n_pos_max`` (positional parameters form a prefix
    of the signature) and by keyword iff ``param_names[i] is not None``
    (None marks positional-only parameters).  This keeps signature-invalid
    calls raising TypeError instead of dispatching on the wrong argument.

    Returns ``(indices, sig_info)`` where ``sig_info`` is
    ``(param_names, n_pos_max, n_required, has_varkw)`` describing the full
    signature.  The C dispatcher uses it to validate the call shape before
    forwarding to an ``__array_function__`` override (the no-override path
    is validated by calling ``implementation`` itself).
    """
    if not relevant_arg_names:
        raise ValueError(
            "tuple-spec dispatcher requires at least one relevant "
            "argument name; got empty tuple")
    sig = inspect.signature(implementation)
    Parameter = inspect.Parameter
    spec = {}  # param name -> parameter index
    param_names = []
    n_pos_max = 0
    n_required = 0
    has_varkw = False
    for pos, (name, param) in enumerate(sig.parameters.items()):
        match param.kind:
            case Parameter.VAR_POSITIONAL:
                raise RuntimeError(
                    f"tuple-spec dispatch does not support implementations "
                    f"with *args; got {implementation.__qualname__}")
            case Parameter.VAR_KEYWORD:
                has_varkw = True
                break
            case Parameter.KEYWORD_ONLY:
                if param.default is Parameter.empty:
                    # validate_call_signature only tracks required positional
                    # parameters; a missing required keyword-only argument
                    # would be forwarded to an override unchecked.
                    raise RuntimeError(
                        f"tuple-spec dispatch does not support required "
                        f"keyword-only parameters; got {name!r} in "
                        f"{implementation.__qualname__}")
                spec[name] = pos           # keyword channel only
                param_names.append(name)
            case Parameter.POSITIONAL_ONLY:
                spec[name] = pos           # positional channel only
                param_names.append(None)
                n_pos_max += 1
            case Parameter.POSITIONAL_OR_KEYWORD:
                spec[name] = pos           # both channels
                param_names.append(name)
                n_pos_max += 1
            case _:  # unreachable: Parameter.kind has no other values
                raise RuntimeError(
                    f"unsupported parameter kind {param.kind!r} for "
                    f"{name!r} in {implementation.__qualname__}")
        # required = contiguous no-default prefix of the positional params
        if (param.default is Parameter.empty and param.kind in (
                Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD)
                and pos == n_required):
            n_required += 1
    resolved = []
    for name in relevant_arg_names:
        if not isinstance(name, str):
            raise TypeError(
                f"tuple-spec dispatcher must contain only strings; "
                f"got {name!r}")
        if name not in spec:
            raise RuntimeError(
                f"relevant arg {name!r} not found in "
                f"{implementation.__qualname__} signature")
        resolved.append(spec[name])
    sig_info = (tuple(param_names), n_pos_max, n_required, has_varkw)
    return tuple(resolved), sig_info


def array_function_dispatch(dispatcher=None, module=None, verify=True,
                            docs_from_dispatcher=False, reduction=None,
                            reduction_defaults=None):
    """Decorator for adding dispatch with the __array_function__ protocol.

    See NEP-18 for example usage.

    Parameters
    ----------
    dispatcher : callable, tuple of str, or None
        If a callable: when called like ``dispatcher(*args, **kwargs)`` with
        arguments from the NumPy function call it returns an iterable of
        array-like arguments to check for ``__array_function__``.

        If a tuple of strings: names of positional/keyword arguments of the
        decorated function that should be checked for ``__array_function__``.

        If ``None``, the first argument is used as the single `like=` argument
        and not passed on.  A function implementing `like=` must call its
        dispatcher with `like` as the first non-keyword argument.
    module : str, optional
        __module__ attribute to set on new function, e.g., ``module='numpy'``.
        By default, module is copied from the decorated function.
    verify : bool, optional
        If True, verify the that the signature of the dispatcher and decorated
        function signatures match exactly: all required and optional arguments
        should appear in order with the same names, but the default values for
        all optional arguments should be ``None``. Only disable verification
        if the dispatcher's signature needs to deviate for some particular
        reason, e.g., because the function has a signature like
        ``func(*args, **kwargs)``.
    docs_from_dispatcher : bool, optional
        If True, copy docs from the dispatcher function onto the dispatched
        function, rather than from the implementation. This is useful for
        functions defined in C, which otherwise don't have docstrings.
    reduction : ufunc or None, optional
        Private: the decorated function is ``reduction.reduce`` under a
        different signature.  Enables an exact-ndarray fast path that
        calls ``reduction.reduce`` directly from C, mapping arguments by
        parameter name (see ``_resolve_reduction_spec``).  Requires a
        tuple-spec dispatcher.
    reduction_defaults : dict or None, optional
        Private: overrides for ``ufunc.reduce`` arguments the public
        signature does not expose (e.g. ``{"dtype": bool}`` for
        ``np.any``/``np.all``).

    Returns
    -------
    Function suitable for decorating the implementation of a NumPy function.

    """
    # exact tuple only (matches C-side PyTuple_CheckExact)
    is_tuple_spec = type(dispatcher) is tuple

    if is_tuple_spec and docs_from_dispatcher:
        raise TypeError(
            "docs_from_dispatcher=True is not supported with a tuple-spec "
            "dispatcher (there is no dispatcher function to copy docs from)")
    if reduction is not None and not is_tuple_spec:
        raise TypeError(
            "reduction= requires a tuple-spec dispatcher")

    def decorator(implementation):
        if is_tuple_spec:
            spec, sig_info = _resolve_relevant_arg_spec(
                implementation, dispatcher)
            if reduction is not None:
                reduction_spec = _resolve_reduction_spec(
                    implementation, reduction, reduction_defaults)
            else:
                reduction_spec = None
            public_api = _ArrayFunctionDispatcher(
                (spec, sig_info, reduction_spec), implementation)
        else:
            if verify:
                if dispatcher is not None:
                    verify_matching_signatures(implementation, dispatcher)
                else:
                    # Using __code__ directly similar to
                    # verify_matching_signatures
                    co = implementation.__code__
                    last_arg = co.co_argcount + co.co_kwonlyargcount - 1
                    last_arg = co.co_varnames[last_arg]
                    if last_arg != "like" or co.co_kwonlyargcount == 0:
                        raise RuntimeError(
                            "__array_function__ expects `like=` to be the "
                            "last argument and a keyword-only argument. "
                            f"{implementation} does not seem to comply.")

            if docs_from_dispatcher and dispatcher.__doc__ is not None:
                doc = inspect.cleandoc(dispatcher.__doc__)
                add_docstring(implementation, doc)

            public_api = _ArrayFunctionDispatcher(dispatcher, implementation)

        functools.update_wrapper(public_api, implementation)

        if not is_tuple_spec and not verify and not getattr(
                implementation, "__text_signature__", None):
            # update_wrapper does not help inspect.signature for
            # implementations with a */** signature; use the dispatcher's.
            public_api.__signature__ = inspect.signature(dispatcher)

        if module is not None:
            public_api.__module__ = module

        ARRAY_FUNCTIONS.add(public_api)

        return public_api

    return decorator


def array_function_from_dispatcher(
        implementation, module=None, verify=True, docs_from_dispatcher=True):
    """Like array_function_dispatcher, but with function arguments flipped."""

    def decorator(dispatcher):
        return array_function_dispatch(
            dispatcher, module, verify=verify,
            docs_from_dispatcher=docs_from_dispatcher)(implementation)
    return decorator
