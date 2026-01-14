from numpy._core.defchararray import __all__, __doc__


def __getattr__(name: str):
    if name == "chararray":
        # Deprecated in NumPy 2.5, 2026-01-07
        import warnings

        warnings.warn(
            (
                "The chararray class is deprecated and will be removed in a future "
                "release. Use an ndarray with a string or bytes dtype instead."
            ),
            DeprecationWarning,
            stacklevel=2,
        )

        from numpy._core.defchararray import chararray

        return chararray

    import numpy._core.defchararray as char

    if (export := getattr(char, name, None)) is not None:
        return export

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    import numpy._core.defchararray as char

    return dir(char)
