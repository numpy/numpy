# Don't use the deprecated NumPy C API. Define this to a fixed version
# instead of NPY_API_VERSION in order not to break compilation for
# released SciPy versions when NumPy introduces a new deprecation. Use
# in setup.py::
#
#   config.add_extension('_name', sources=['source_fname'], **numpy_nodepr_api)
#
numpy_nodepr_api = dict(
    define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_9_API_VERSION")]
)


def import_file(folder, module_name):
    """Import a file directly, avoiding importing scipy"""
    import importlib
    import pathlib

    fname = pathlib.Path(folder) / f'{module_name}.py'
    spec = importlib.util.spec_from_file_location(module_name, str(fname))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
