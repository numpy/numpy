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

# ufunc.reduce positional argument order and defaults, used to express
# reduction=<ufunc> as a forward spec (see _resolve_forward_spec).
_REDUCE_SLOT_NAMES = ("a", "axis", "dtype", "out", "keepdims", "initial", "where")
_REDUCE_DEFAULTS = {
    "axis": None, "dtype": None, "out": None,
    "keepdims": False, "initial": _NoValue, "where": True,
}


def _signature_items(implementation):
    """Yield ``(name, kind, default)`` for ``implementation``'s parameters.

    Reads the code object directly (``inspect.signature`` is 10x slower
    and would be paid per decoration at import time); wrapped
    implementations fall back to ``inspect.signature``.
    """
    Parameter = inspect.Parameter
    if (hasattr(implementation, "__signature__")
            or hasattr(implementation, "__wrapped__")
            or not hasattr(implementation, "__code__")):
        for name, param in inspect.signature(
                implementation).parameters.items():
            yield name, param.kind, param.default
        return
    code = implementation.__code__
    names = code.co_varnames
    n_posonly = code.co_posonlyargcount
    n_positional = code.co_argcount
    n_kwonly = code.co_kwonlyargcount
    defaults = implementation.__defaults__ or ()
    kwdefaults = implementation.__kwdefaults__ or {}
    first_default = n_positional - len(defaults)
    for i in range(n_positional):
        kind = (Parameter.POSITIONAL_ONLY if i < n_posonly
                else Parameter.POSITIONAL_OR_KEYWORD)
        default = (defaults[i - first_default] if i >= first_default
                   else Parameter.empty)
        yield names[i], kind, default
    if code.co_flags & inspect.CO_VARARGS:
        yield (names[n_positional + n_kwonly], Parameter.VAR_POSITIONAL,
               Parameter.empty)
    for i in range(n_positional, n_positional + n_kwonly):
        yield (names[i], Parameter.KEYWORD_ONLY,
               kwdefaults.get(names[i], Parameter.empty))
    if code.co_flags & inspect.CO_VARKEYWORDS:
        index = (n_positional + n_kwonly
                 + bool(code.co_flags & inspect.CO_VARARGS))
        yield names[index], Parameter.VAR_KEYWORD, Parameter.empty


def _resolve_forward_spec(implementation, target, slot_names,
                          defaults_override):
    """Map ``implementation``'s parameters onto a direct call of ``target``.

    ``slot_names`` lists the target's argument slots in call order (a
    leading ``"*"`` marks trailing keyword-only slots).  Parameters are
    matched to slots by name; a parameter without a slot declines the
    fast path when passed.  Defaults come from ``defaults_override`` or
    the parameter's own default; ``np._NoValue`` defaults require an
    explicit override.

    Returns ``(target, slots, defaults, kwnames, n_slots, out_slot,
    where_slot, novalue_mask, required_mask)`` where ``slots[i]`` is the
    target-slot of the i-th parameter (-1: no slot).
    """
    names = [n.lstrip("*") for n in slot_names]
    n_kw = sum(1 for n in slot_names if n.startswith("*"))
    if n_kw and not all(n.startswith("*") for n in slot_names[-n_kw:]):
        raise RuntimeError(
            f"keyword slots must be trailing in {slot_names!r}")
    kwnames = tuple(names[-n_kw:]) if n_kw else None
    slot_of = {name: i for i, name in enumerate(names)}
    if names[0] != "a":
        raise RuntimeError(
            f"first target slot must be 'a', got {names[0]!r}")

    Parameter = inspect.Parameter
    slots = []
    defaults = [None] * len(names)
    novalue_mask = 0
    required_mask = 0
    filled = {0}
    for name, kind, default in _signature_items(implementation):
        if kind not in (Parameter.POSITIONAL_ONLY,
                        Parameter.POSITIONAL_OR_KEYWORD,
                        Parameter.KEYWORD_ONLY):
            raise RuntimeError(
                f"forward fast path cannot map parameter {name!r} of "
                f"{implementation.__qualname__}")
        slot = slot_of.get(name, -1)
        slots.append(slot)
        if slot <= 0:
            continue
        filled.add(slot)
        if defaults_override and name in defaults_override:
            override = defaults_override[name]
            if default is Parameter.empty:
                # the fast path would succeed where the wrapper raises
                raise RuntimeError(
                    f"forward default for required parameter {name!r} of "
                    f"{implementation.__qualname__}")
            if default is not _NoValue and default != override:
                # the fast path would pass `override` where the wrapper
                # passes `default`, so the two would disagree
                raise RuntimeError(
                    f"forward default {override!r} for {name!r} of "
                    f"{implementation.__qualname__} contradicts its "
                    f"signature default {default!r}")
            defaults[slot] = override
        elif default is _NoValue:
            raise RuntimeError(
                f"parameter {name!r} of {implementation.__qualname__} "
                f"defaults to np._NoValue; forward spec needs an explicit "
                f"default override")
        elif default is Parameter.empty:
            required_mask |= 1 << slot
        else:
            defaults[slot] = default
        if default is _NoValue:
            novalue_mask |= 1 << slot
    if defaults_override:
        for name, value in defaults_override.items():
            if name not in slot_of:
                raise KeyError(name)
            if slot_of[name] == 0:
                # a default for the array slot would be silently dead
                raise RuntimeError(
                    f"forward default for {name!r} (the array slot) has "
                    f"no effect")
            defaults[slot_of[name]] = value
            filled.add(slot_of[name])
    missing = set(range(len(names))) - filled
    if missing:
        raise RuntimeError(
            f"target slots {sorted(missing)} of {slot_names!r} have no "
            f"public parameter and no default override")
    return (target, tuple(slots), tuple(defaults), kwnames, len(names),
            slot_of.get("out", -1), slot_of.get("where", -1),
            novalue_mask, required_mask)


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
    dispatcher : function, tuple, or None
        Either a dispatcher function that returns a single sequence-like
        object of all relevant arguments (same signature as the
        implementation, except default values), a ``(spec, sig_info,
        forward_spec)`` tuple built by ``array_function_dispatch`` from
        a tuple-spec, or ``None`` for a ``like=`` dispatcher (called with
        ``like`` as the first positional argument).
    implementation : function
        Function that implements the operation on NumPy arrays without
        overrides.  Arguments passed calling the ``_ArrayFunctionDispatcher``
        will be forwarded to this (and the ``dispatcher``) as if using
        ``*args, **kwargs``.

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
    which they should be called.
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
    signature; rejects an empty spec, ``*args``, and unknown names.

    Returns ``(indices, sig_info)`` with ``sig_info =
    (param_names, n_pos_max, n_required, has_varkw)``.  Index ``i`` may be
    matched positionally iff ``i < n_pos_max`` and by keyword iff
    ``param_names[i] is not None``, so signature-invalid calls raise
    TypeError instead of dispatching on the wrong argument.
    """
    if not relevant_arg_names:
        raise ValueError(
            "tuple-spec dispatcher requires at least one relevant "
            "argument name; got empty tuple")
    Parameter = inspect.Parameter
    spec = {}  # param name -> parameter index
    param_names = []
    n_pos_max = 0
    n_required = 0
    has_varkw = False
    for pos, (name, kind, default) in enumerate(
            _signature_items(implementation)):
        match kind:
            case Parameter.VAR_POSITIONAL:
                raise RuntimeError(
                    f"tuple-spec dispatch does not support implementations "
                    f"with *args; got {implementation.__qualname__}")
            case Parameter.VAR_KEYWORD:
                has_varkw = True
                break
            case Parameter.KEYWORD_ONLY:
                if default is Parameter.empty:
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
        # required = contiguous no-default prefix of the positional params
        if (default is Parameter.empty and kind in (
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
                            reduction_defaults=None, forward=None,
                            forward_defaults=None):
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
        Sugar for ``forward=(reduction.reduce, <reduce slots>)``
        with ``ufunc.reduce``'s defaults.  Requires a tuple-spec
        dispatcher.
    reduction_defaults : dict or None, optional
        Overrides for ``ufunc.reduce`` arguments the public
        signature does not expose (e.g. ``{"dtype": bool}`` for
        ``np.any``/``np.all``).
    forward : (callable, tuple of str) or None, optional
        The decorated function is ``callable`` under a different
        signature; the tuple lists the target's argument slots in call
        order (a leading ``"*"`` marks keyword-only slots).  Enables an
        exact-ndarray fast path that calls ``callable`` (typically an
        ``ndarray`` method, replacing a ``_wrapfunc``-style wrapper)
        directly from C (see ``_resolve_forward_spec``).  Requires a
        tuple-spec dispatcher; mutually exclusive with ``reduction``.
    forward_defaults : dict or None, optional
        Overrides for target arguments whose public default is
        ``np._NoValue`` or that the public signature does not expose.

    Returns
    -------
    Function suitable for decorating the implementation of a NumPy function.

    """
    is_tuple_spec = type(dispatcher) is tuple

    if is_tuple_spec and docs_from_dispatcher:
        raise TypeError(
            "docs_from_dispatcher=True is not supported with a tuple-spec "
            "dispatcher (there is no dispatcher function to copy docs from)")
    if reduction is not None and not is_tuple_spec:
        raise TypeError(
            "reduction= requires a tuple-spec dispatcher")
    if reduction is None and reduction_defaults is not None:
        raise TypeError(
            "reduction_defaults= requires reduction=")
    if forward is not None:
        if not is_tuple_spec:
            raise TypeError("forward= requires a tuple-spec dispatcher")
        if reduction is not None:
            raise TypeError("forward= and reduction= are mutually exclusive")
    elif forward_defaults is not None:
        raise TypeError("forward_defaults= requires forward=")

    def decorator(implementation):
        if is_tuple_spec:
            spec, sig_info = _resolve_relevant_arg_spec(
                implementation, dispatcher)
            if reduction is not None:
                # reduction=<ufunc> is sugar for forwarding to ufunc.reduce
                defaults = dict(_REDUCE_DEFAULTS)
                if reduction_defaults:
                    defaults.update(reduction_defaults)
                forward_spec = _resolve_forward_spec(
                    implementation, reduction.reduce, _REDUCE_SLOT_NAMES,
                    defaults)
            elif forward is not None:
                forward_spec = _resolve_forward_spec(
                    implementation, forward[0], forward[1], forward_defaults)
            else:
                forward_spec = None
            public_api = _ArrayFunctionDispatcher(
                (spec, sig_info, forward_spec), implementation)
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
