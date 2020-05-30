"""
Build/Install NumPy on Windows with clang-cl using (powershell syntax)

Short:
python setup.py build --compiler=clang-cl install

Full:
python setup.py config --compiler=clang-cl build_clib `
                       --compiler=clang-cl build_ext `
                       --compiler=clang-cl install

Develop mode:
python setup.py build_ext -i --compiler=clang-cl develop

Run tests:
python runtests.py --compiler=clang-cl
"""

import os
import subprocess

try:
    from distutils._msvccompiler import MSVCCompiler, _find_exe
except ImportError:
    # Dummy to pass import test on non-Windows
    MSVCCompiler = object

from .system_info import platform_bits
from .msvccompiler import _merge


class ClangCL(MSVCCompiler):
    """
    A modified clang-cl compiler compatible with an MSVC-built Python.
    """

    compiler_type = "clang_cl"
    compiler_cxx = "clang-cl"

    def __init__(self, verbose=0, dry_run=0, force=0):
        super().__init__(verbose, dry_run, force)

    def initialize(self):
        # The 'lib' and 'include' variables may be overwritten
        # by MSVCCompiler.initialize, so save them for later merge.
        environ_lib = os.getenv("lib", "")
        environ_include = os.getenv("include", "")
        MSVCCompiler.initialize(self)

        self.cc = _find_exe("clang-cl.exe")
        if self.cc is None:
            raise FileNotFoundError(
                "Unable to locate clang-cl.exe. It should " "be on the path"
            )
        for opt in ("/GL", "/GL-"):
            if opt in self.compile_options:
                self.compile_options.remove(opt)
        self.compile_options.remove("/Ox")
        self.compile_options.extend(
            [
                "/O2",
                "/GS-",
                "-Wno-visibility",
                "-Wno-logical-op-parentheses",
                "-Wno-microsoft-include",
                "-Wno-shift-op-parentheses",
            ]
        )

        if platform_bits == 32:
            self.compile_options += ['/arch:SSE2']
            self.compile_options_debug += ['/arch:SSE2']

        # Merge current and previous values of 'lib' and 'include'
        os.environ["lib"] = _merge(environ_lib, os.environ["lib"])
        os.environ["include"] = _merge(environ_include, os.environ["include"])
        clang_base = os.path.split(self.cc)[0]
        clang_version_cmd = [self.cc, "--version"]
        clang_full_ver = subprocess.check_output(clang_version_cmd).decode()
        clang_version = None
        if "version" in clang_full_ver:
            clang_ver_string = clang_full_ver[clang_full_ver.find("version") + 7:]
            clang_version = clang_ver_string.split("\n")[0].strip()
        clang_command_and_return = f"""\
Running the command \n\n{' '.join(clang_version_cmd)}\n\n which returned\
\n\n{clang_full_ver}
"""
        # Forbid compiling 32 on 64 and vice versa
        target = "i686" if platform_bits == 32 else "x86_64"
        if target not in clang_full_ver:
            raise RuntimeError(
                f"clang-cl must target {target} when building on "
                f"{platform_bits}-bit windows. {clang_command_and_return}"
            )
        if clang_version is None:
            raise RuntimeError(
                f"clang_version could not be detected from the version "
                f"string. {clang_command_and_return}"
            )
        clang_incl = ["..", "lib", "clang", clang_version, "include"]
        clang_incl = os.path.abspath(os.path.join(clang_base, *clang_incl))
        assert os.path.exists(clang_incl)
        self.include_dirs.insert(0, clang_incl)

        # Need to block MS includes except those from Visual Studio 14.0
        # since they include intrinsics that break clang
        def retain_include(path):
            blocked = ("Windows Kits", "MSVC", "2019")
            keep = True
            for block in blocked:
                keep = keep and block not in path
            return keep

        include_dirs = [
            path for path in self.include_dirs if retain_include(path)
        ]
        existing = set()
        include_dirs = [
            path
            for path in include_dirs
            if not (path in existing or existing.add(path))
        ]
        self.include_dirs = include_dirs
        if platform_bits == 32:
            clang_lib = ["..", "lib", "clang", clang_version, "lib", "windows"]
            clang_lib = os.path.abspath(os.path.join(clang_base, *clang_lib))
            if not os.path.exists(clang_lib):
                raise RuntimeError()
            self.add_library_dir(clang_lib)
            if not os.path.exists(os.path.join(clang_lib,"clang_rt.builtins-i386.lib")):
                raise RuntimeError()
            self.add_library("clang_rt.builtins-i386")

