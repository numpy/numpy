
# http://www.pgroup.com

import os
import sys

from cpuinfo import cpu
from fcompiler import FCompiler

class PGroupFCompiler(FCompiler):

    compiler_type = 'pg'
    version_pattern =  r'\s*pg(f77|f90|hpf) (?P<version>[\d.-]+).*'

    executables = {
        'version_cmd'  : ["pgf77", "-V 2>/dev/null"],
        'compiler_f77' : ["pgf77"],
        'compiler_fix' : ["pgf90", "-Mfixed"],
        'compiler_f90' : ["pgf90"],
        'linker_so'    : ["pgf90","-shared"],
        'archiver'     : ["ar", "-cr"],
        'ranlib'       : ["ranlib"]
        }
    pic_flags = ['-fpic']
    module_dir_switch = '-module '
    module_include_switch = '-I'

    def get_flags(self):
        opt = ['-Minform=inform','-Mnosecond_underscore']
        return self.pic_flags + opt
    def get_flags_opt(self):
        return ['-fast']
    def get_flags_debug(self):
        return ['-g']

if __name__ == '__main__':
    from distutils import log
    log.set_verbosity(2)
    from fcompiler import new_fcompiler
    compiler = new_fcompiler(compiler='pg')
    compiler.customize()
    print compiler.get_version()
