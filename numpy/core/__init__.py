import warnings

from numpy import _core


def __getattr__(attr):

    warnings.warn(
        f"`np.core` has been made officially private and renamed to "
        "`np._core`. Accessing `_core` directly is discouraged, you "
        "should use a public module instead that exports the attribute "
        "in question. If you still would like to access an attribute "
        f"from it, please use `np._core.{attr}`.",
        DeprecationWarning, 
        stacklevel=2
    )
    
    return getattr(_core, attr)
