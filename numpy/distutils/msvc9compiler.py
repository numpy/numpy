import os
import distutils.msvc9compiler
from distutils.msvc9compiler import *


class MSVCCompiler(distutils.msvc9compiler.MSVCCompiler):
    def __init__(self, verbose=0, dry_run=0, force=0):
        distutils.msvc9compiler.MSVCCompiler.__init__(self, verbose, dry_run, force)

    def initialize(self, plat_name=None):
        environ_lib = os.getenv('lib')
        environ_include = os.getenv('include')
        distutils.msvc9compiler.MSVCCompiler.initialize(self, plat_name)
        if environ_lib is not None:
            os.environ['lib'] = environ_lib + os.environ['lib']
        if environ_include is not None:
            os.environ['include'] = environ_include + os.environ['include']

    def manifest_setup_ldargs(self, output_filename, build_temp, ld_args):
        ld_args.append('/MANIFEST')
        distutils.msvc9compiler.MSVCCompiler.manifest_setup_ldargs(self,
                                                                   output_filename,
                                                                   build_temp, ld_args)
