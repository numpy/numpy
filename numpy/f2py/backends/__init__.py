def get_backend(name):
    if name == "meson":
        from .meson_backend import MesonBackend

        return MesonBackend
    elif name == "distutils":
        from .distutils_backend import DistutilsBackend

        return DistutilsBackend
    else:
        raise ValueError(f"Unknown backend: {name}")
