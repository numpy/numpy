from numpy.f2py.backends.backend import Backend

from numpy.distutils.core import setup, Extension
from numpy.distutils.system_info import get_info
from numpy.distutils.misc_util import dict_append
import os
import sys
import shutil
import warnings


class DistutilsBackend(Backend):
    def __init__(sef, *args, **kwargs):
        warnings.warn(
            "distutils has been deprecated since NumPy 2.16."
            "Use the MesonBackend instead, or generate wrappers"
            "without -c and use a custom build script",
            np.VisibleDeprecationWarning, stacklevel=2)
        super().__init__(*args, **kwargs)

    def compile(self, sources, extra_objects, build_dir):
        num_info = {}
        if num_info:
            self.include_dirs.extend(num_info.get("include_dirs", []))
        ext_args = {
            "name": self.modulename,
            "sources": sources,
            "include_dirs": self.include_dirs,
            "library_dirs": self.library_dirs,
            "libraries": self.libraries,
            "define_macros": self.define_macros,
            "undef_macros": self.undef_macros,
            "extra_objects": extra_objects,
            "f2py_options": self.f2py_flags,
        }

        if self.sysinfo_flags:
            for n in self.sysinfo_flags:
                i = get_info(n)
                if not i:
                    print(
                        f"No {repr(n)} resources found in system (try `f2py --help-link`)"
                    )
                dict_append(ext_args, **i)

        ext = Extension(**ext_args)

        sys.argv = [sys.argv[0]] + self.setup_flags
        sys.argv.extend(
            [
                "build",
                "--build-temp",
                build_dir,
                "--build-base",
                build_dir,
                "--build-platlib",
                ".",
                "--disable-optimization",
            ]
        )

        if self.fc_flags:
            sys.argv.extend(["config_fc"] + self.fc_flags)
        if self.flib_flags:
            sys.argv.extend(["build_ext"] + self.flib_flags)

        setup(ext_modules=[ext])

        if self.remove_build_dir and os.path.exists(build_dir):
            print(f"Removing build directory {build_dir}")
            shutil.rmtree(build_dir)
