import os
import distutils.msvccompiler
from distutils.msvccompiler import *

from .system_info import platform_bits


class MSVCCompiler(distutils.msvccompiler.MSVCCompiler):
    def __init__(self, verbose=0, dry_run=0, force=0):
        distutils.msvccompiler.MSVCCompiler.__init__(self, verbose, dry_run, force)

    def initialize(self, plat_name=None):
        environ_lib = os.getenv('lib')
        environ_include = os.getenv('include')
        distutils.msvccompiler.MSVCCompiler.initialize(self, plat_name)
        if environ_lib is not None:
            os.environ['lib'] = environ_lib + os.environ['lib']
        if environ_include is not None:
            os.environ['include'] = environ_include + os.environ['include']
        if platform_bits == 32:
            # msvc9 building for 32 bits requires SSE2 to work around a
            # compiler bug.
            self.compile_options += ['/arch:SSE2']
            self.compile_options_debug += ['/arch:SSE2']
