"""
Build/Install NumPy on Windows with clang-cl using (powershell syntax)

Short:
python setup.py build --compiler=clang-cl install

Full:
python setup.py config --compiler=clang-cl `
                build_clib --compiler=clang-cl `
                build_ext --compiler=clang-cl `
                install

Develop mode:
python setup.py build_ext -i --compiler=clang-cl develop

Run tests:
python runtests.py --compiler=clang-cl
"""

import os
import subprocess
from .system_info import platform_bits
from .msvccompiler import _merge

try:
    from distutils._msvccompiler import MSVCCompiler, _find_exe
except ImportError:
    # Dummy to pass import test on non-Windows
    class MSVCCompiler:
        def __init__(self, verbose=0, dry_run=0, force=0):
            raise NotImplementedError(
                "ClangLC can only be run on Windows when MSVCCompiler can "
                "be imported from distutils._msvccompiler"
            )


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
                "Unable to locate clang-cl.exe. It should be on the path"
            )
        for opt in ("/GL", "/GL-", "/Ox"):
            if opt in self.compile_options:
                self.compile_options.remove(opt)
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

        # Merge current and previous values of 'lib' and 'include'
        os.environ["lib"] = _merge(environ_lib, os.environ["lib"])
        os.environ["include"] = _merge(environ_include, os.environ["include"])
        # Get version information about clang which is needed to find
        # the correct include and lib directories
        clang_base = os.path.split(self.cc)[0]
        clang_version_cmd = [self.cc, "--version"]
        # Get version or let the user know if there was an issue
        try:
            clang_full_ver = subprocess.check_output(
                clang_version_cmd
            ).decode()
        except WindowsError:
            raise WindowsError(
                "clang-cl.exe could not be found. It must be on your path."
            )
        clang_version = ""
        if "version" in clang_full_ver:
            loc = clang_full_ver.find("version")
            clang_ver_string = clang_full_ver[loc + 7 :]
            clang_version = clang_ver_string.split("\n")[0].strip()
        clang_command_and_return = (
            f"Running the command \n\n{' '.join(clang_version_cmd)}\n\n "
            f"which returned \n\n{clang_full_ver}"
        )

        if not clang_version:
            raise RuntimeError(
                f"The clang version could not be detected from the version "
                f"string. {clang_command_and_return}"
            )

        # Forbid compiling 32 on 64-bit LLVM and vice versa
        target = "i686" if platform_bits == 32 else "x86_64"
        if target not in clang_full_ver:
            raise RuntimeError(
                f"clang-cl must target {target} when building on "
                f"{platform_bits}-bit windows. {clang_command_and_return}"
            )
        # Add the include directory for clang
        clang_incl = ["..", "lib", "clang", clang_version, "include"]
        clang_incl = os.path.abspath(os.path.join(clang_base, *clang_incl))
        if not os.path.exists(clang_incl):
            raise RuntimeError(
                f"{clang_incl} could not be found. This directory contains "
                f"includes required by clang-cl."
            )
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
        # Deduplicate the included dirs but keep them in order
        existing = set()
        include_dirs = [
            path
            for path in include_dirs
            if not (path in existing or existing.add(path))
        ]
        self.include_dirs = include_dirs

        # Add the builtins library
        builtins_target = "x86_64" if platform_bits == 64 else "i386"
        builtins_lib = f"clang_rt.builtins-{builtins_target}"
        clang_lib = ["..", "lib", "clang", clang_version, "lib", "windows"]
        clang_lib = os.path.abspath(os.path.join(clang_base, *clang_lib))
        if not os.path.exists(os.path.join(clang_lib, builtins_lib + ".lib")):
            raise RuntimeError(
                f"{builtins_lib}.lib could not be found in {clang_lib}. This "
                f"library is supplies built-ins that clang-cl uses that are "
                f"not part of MSVC."
            )
        self.add_library_dir(clang_lib)
        self.add_library(builtins_lib)

        # Enable/require SSE2 for i686
        if platform_bits == 32:
            self.compile_options += ["/arch:SSE2"]
            self.compile_options_debug += ["/arch:SSE2"]
