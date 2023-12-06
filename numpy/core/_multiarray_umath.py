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
                For cibuildwheel for example, this may be achieved by using
                the flag to pip:
                    CIBW_BUILD_FRONTEND: pip; args: --no-build-isolation
                installing NumPy with:
                    pip install --pre --extra-index-url https://pypi.anaconda.org/scientific-python-nightly-wheels/simple
                in the `CIBW_BEFORE_BUILD` step.  Please compare with the
                solutions e.g. in astropy or matplotlib for how to make this
                conditional for nightly wheel builds using expressions.
                If you do not worry about using pre-releases of all
                dependencies, you can also use `--pre --extra-index-url` in the
                build frontend (instead of build isolation).
                This will become unnecessary as soon as NumPy 2.0 is released.

                If your dependencies have the issue, check whether they
                have nightly wheels build against NumPy 2.0.

                pybind11 note: You may see this message if using pybind11,
                this is not problematic at pre-release time
                it indicates the need for a new pybind11 release.

                """)
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
