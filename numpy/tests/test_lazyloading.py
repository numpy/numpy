import sys
import importlib
from importlib.util import LazyLoader, find_spec, module_from_spec
import pytest


@pytest.mark.filterwarnings("ignore:The NumPy")
def test_lazy_load():
    # gh-XXXX. lazyload doesn't import submodule names into the namespace
    # muck with sys.modules to test the importing system
    old_numpy = sys.modules.pop("numpy")

    numpy_modules = {}
    for mod_name, mod in list(sys.modules.items()):
        if mod_name[:6] == "numpy.":
            numpy_modules[mod_name] = mod
            sys.modules.pop(mod_name)

    try:
        # create lazy load of numpy as np
        spec = find_spec("numpy")
        module = module_from_spec(spec)
        sys.modules["numpy"] = module
        loader = LazyLoader(spec.loader)
        loader.exec_module(module)
        np = module

        # test a subpackage import
        from numpy.lib import recfunctions

        # test triggering the import of the package
        np.ndarray

    finally:
        if old_numpy:
            sys.modules["numpy"] = old_numpy
            sys.modules.update(numpy_modules)
