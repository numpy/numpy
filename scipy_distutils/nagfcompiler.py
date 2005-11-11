import os
import sys

from cpuinfo import cpu
from fcompiler import FCompiler

class NAGFCompiler(FCompiler):

    compiler_type = 'nag'
    version_pattern =  r'NAGWare Fortran 95 compiler Release (?P<version>[^\s]*)'

    executables = {
        'version_cmd'  : ["f95", "-V"],
        'compiler_f77' : ["f95", "-fixed"],
        'compiler_fix' : ["f95", "-fixed"],
        'compiler_f90' : ["f95"],
        'linker_so'    : ["f95"],
        'archiver'     : ["ar", "-cr"],
        'ranlib'       : ["ranlib"]
        }

    def get_flags_linker_so(self):
        if sys.platform=='darwin':
            return ['-unsharedf95','-Wl,-bundle,-flat_namespace,-undefined,suppress']
        return ["-Wl,shared"]
    def get_flags_opt(self):
        return ['-O4']
    def get_flags_arch(self):
        return ['-target=native']
    def get_flags_debug(self):
        return ['-g','-gline','-g90','-nan','-C']

if __name__ == '__main__':
    from distutils import log
    log.set_verbosity(2)
    from fcompiler import new_fcompiler
    compiler = new_fcompiler(compiler='nag')
    compiler.customize()
    print compiler.get_version()
