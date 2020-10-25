
import importlib.util
import os.path
import sys

def _get_version():
    numpy_version_py = os.path.join(os.path.dirname(__file__), '..', 'numpy', 'version.py')
    spec = importlib.util.spec_from_file_location('f2py._numpy_version', numpy_version_py)
    module = importlib.util.module_from_spec(spec)
    sys.modules['f2py._numpy_version'] = module
    spec.loader.exec_module(module)
    return module.version

numpy_version = _get_version()
