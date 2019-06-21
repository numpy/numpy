from __future__ import division, absolute_import, print_function

import platform
from distutils.ccompiler import CCompiler
from numpy.distutils.ccompiler import simple_version_match

if platform.system() != 'Windows':
    raise NotImplementedError('This is only for clang-cl')

import os
from numpy.distutils.exec_command import find_executable
from distutils._msvccompiler import MSVCCompiler as _MSVCCompiler, _find_exe

from .system_info import platform_bits


def _merge(old, new):
    """Concatenate two environment paths avoiding repeats.

    Here `old` is the environment string before the base class initialize
    function is called and `new` is the string after the call. The new string
    will be a fixed string if it is not obtained from the current environment,
    or the same as the old string if obtained from the same environment. The aim
    here is not to append the new string if it is already contained in the old
    string so as to limit the growth of the environment string.

    Parameters
    ----------
    old : string
        Previous environment string.
    new : string
        New environment string.

    Returns
    -------
    ret : string
        Updated environment string.

    """
    if new in old:
        return old
    if not old:
        return new

    # Neither new nor old is empty. Give old priority.
    return ';'.join([old, new])


class ClangCL(_MSVCCompiler):
    """
    A modified clang-cl compiler compatible with an MSVC-built Python.
    """
    compiler_type = 'clang_cl'
    compiler_cxx = 'clang-cl'

    def __init__(self, verbose=0, dry_run=0, force=0):
        _MSVCCompiler.__init__(self, verbose, dry_run, force)


    def initialize(self):
        # The 'lib' and 'include' variables may be overwritten
        # by MSVCCompiler.initialize, so save them for later merge.
        environ_lib = os.getenv('lib', '')
        environ_include = os.getenv('include', '')
        _MSVCCompiler.initialize(self)
        msvc_path = os.path.split(self.cc)[0]
        self.cc = find_executable('clang-cl.exe')
        if self.cc is None:
            raise FileNotFoundError('Unable to locate clang-cl.exe. It should '
                                    'be on the path')

        self.compile_options.remove('/GL')
        self.compile_options.remove('/Ox')
        self.compile_options.extend(['/O2', '/GS-', '-Wno-visibility'])
        ldflags = [
            '/nologo', '/INCREMENTAL:NO'
        ]
        if not self._vcruntime_redist:
            ldflags.extend(('/nodefaultlib:libucrt.lib', 'ucrt.lib'))

        ldflags_debug = [
            '/nologo', '/INCREMENTAL:NO', '/DEBUG:FULL'
        ]

        self.ldflags_exe = [*ldflags, '/MANIFEST:EMBED,ID=1']
        self.ldflags_shared = [*ldflags, '/DLL', '/MANIFEST:EMBED,ID=2',
                               '/MANIFESTUAC:NO']
        self.ldflags_static = [*ldflags]

        self._ldflags = {
            (CCompiler.EXECUTABLE, None): self.ldflags_exe,
            (CCompiler.EXECUTABLE, False): self.ldflags_exe,
            (CCompiler.EXECUTABLE, True): self.ldflags_exe_debug,
            (CCompiler.SHARED_OBJECT, None): self.ldflags_shared,
            (CCompiler.SHARED_OBJECT, False): self.ldflags_shared,
            (CCompiler.SHARED_OBJECT, True): self.ldflags_shared_debug,
            (CCompiler.SHARED_LIBRARY, None): self.ldflags_static,
            (CCompiler.SHARED_LIBRARY, False): self.ldflags_static,
            (CCompiler.SHARED_LIBRARY, True): self.ldflags_static_debug,
        }

        # Merge current and previous values of 'lib' and 'include'
        os.environ['lib'] = _merge(environ_lib, os.environ['lib'])
        os.environ['include'] = _merge(environ_include, os.environ['include'])
        clang_base = os.path.split(self.cc)[0]
        # TODO: Extract clang version
        clang_version = '8.0.0'
        clang_incl = ['..', 'lib', 'clang', clang_version, 'include']
        clang_incl = os.path.abspath(os.path.join(clang_base, *clang_incl))
        self.include_dirs.insert(0, clang_incl)

        def tokenize_path(path):
            path, end = os.path.split(path)
            tokens = []
            while end:
                tokens.append(end)
                path, end = os.path.split(path)
            tokens.append(path)
            return tokens[::-1]

        msvc_path = tokenize_path(msvc_path)
        best_score = -1
        msvc_include = None
        for path in self.include_dirs:
            if not os.path.split(path)[1] == 'include':
                continue
            tokenized = tokenize_path(path)
            score = sum([a == b for a, b in zip(msvc_path, tokenized)])
            if score > best_score:
                best_score = score
                msvc_include = path
        if msvc_include:
            self.include_dirs.remove(msvc_include)
        # msvc9 building for 32 bits requires SSE2 to work around a
        # compiler bug.
        if platform_bits == 32:
            self.compile_options += ['/arch:SSE2']
            self.compile_options_debug += ['/arch:SSE2']
