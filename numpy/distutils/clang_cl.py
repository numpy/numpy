"""
Build/Install NumPy on Windows with clang-cl using (powershell syntax)

python setup.py config --compiler=clang-cl build_clib `
                       --compiler=clang-cl build_ext `
                       --compiler=clang-cl install
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
        out = subprocess.check_output(clang_version_cmd).decode()
        clang_version = None
        if out.find("version"):
            out = out[out.find("version") + 7:]
            clang_version = out.split("\n")[0].strip()
        if clang_version is None:
            raise RuntimeError(
                "clang_version could not be detected from the version string "
                f"returned when running\n\n{' '.join(clang_version_cmd)}\n\n "
                f"which returned\n\n{out}"
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

