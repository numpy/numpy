from numpy._core import _multiarray_umath
from numpy import ufunc

for item in _multiarray_umath.__dir__():
    # ufuncs appear in pickles with a path in numpy.core._multiarray_umath
    # and so must import from this namespace without warning or error
    attr = getattr(_multiarray_umath, item)
    if isinstance(attr, ufunc):
        globals()[item] = attr


def __getattr__(attr_name):
    from numpy._core import _multiarray_umath
    from ._utils import _raise_warning

    if attr_name in {"_ARRAY_API", "_UFUNC_API"}:
        from numpy.version import short_version, release
        import textwrap
        import traceback
        import sys

        msg = textwrap.dedent(f"""
            A module that was compiled using NumPy 1.x cannot be run in
            NumPy {short_version} as it may crash. To support both 1.x and 2.x
            versions of NumPy, modules must be compiled against NumPy 2.0.

            If you are a user of the module, the easiest solution will be to
            either downgrade NumPy or update the failing module (if available).

            """)
        if not release and short_version.startswith("2.0.0"):
            # TODO: Can remove this after the release.
            msg += textwrap.dedent("""\
                NOTE: When testing against pre-release versions of NumPy 2.0
                or building nightly wheels for it, it is necessary to ensure
                the NumPy pre-release is used at build time.
                The main way to ensure this is using no build isolation
                and installing dependencies manually with NumPy.

                If your dependencies have the issue, check whether they
                build nightly wheels build against NumPy 2.0.

                pybind11 note: If you see this message and do not see
                any errors raised, it's possible this is due to a
                package using an old version of pybind11 that should be
                updated.

                """)
        msg += "Traceback (most recent call last):"
        for line in traceback.format_stack()[:-1]:
            if "frozen importlib" in line:
                continue
            msg += line
        # Only print the message.  This has two reasons (for now!):
        # 1. Old NumPy replaced the error here making it never actually show
        #    in practice, thus raising alone would not be helpful.
        # 2. pybind11 simply reaches into NumPy internals and requires a
        #    new release that includes the fix. That is missing as of 2023-11.
        #    But, it "conveniently" ignores the ABI version.
        sys.stderr.write(msg)

    ret = getattr(_multiarray_umath, attr_name, None)
    if ret is None:
        raise AttributeError(
            "module 'numpy.core._multiarray_umath' has no attribute "
            f"{attr_name}")
    _raise_warning(attr_name, "_multiarray_umath")
    return ret


del _multiarray_umath, ufunc
