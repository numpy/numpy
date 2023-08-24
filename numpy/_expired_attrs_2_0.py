"""
Dict of expired attributes that are discontinued since 2.0 release.
Each item is associated with a migration note.
"""

__expired_attributes__ = {
    "geterrobj": "Use context manager `with np.errstate()` instead.",
    "seterrobj": "Use context manager `with np.errstate()` instead.",
    "cast": "Use `np.asarray(arr, dtype=dtype)` instead.",
    "source": "Use `inspect.getsource` instead.",
    "lookfor":  "Search NumPy's documentation directly.",
    "who": "Use IDE variable explorer or `locals()` instead.",
    "fastCopyAndTranspose": "Use `arr.T.copy()` instead.",
    "set_numeric_ops": 
        "For the general case, use `PyUFunc_ReplaceLoopBySignature`. "
        "For ndarray subclasses, define the ``__array_ufunc__`` method "
        "and override the relevant ufunc.",
    "NINF": "Use `-np.inf` instead.",
    "PINF": "Use `np.inf` instead.",
    "NZERO": "Use `-0.0` instead.",
    "PZERO": "Use `0.0` instead.",
    "add_newdoc": 
        "It's and internal function and doesn't have a replacement.",
    "add_docstring": 
        "It's and internal function and doesn't have a replacement.",
    "add_newdoc_ufunc": 
        "It's and internal function and doesn't have a replacement.",
    "compat": "There's no replacement, as Python 2 is no longer supported.",
    "safe_eval": "Use `ast.literal_eval` instead."
}
