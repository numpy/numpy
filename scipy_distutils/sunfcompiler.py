import os
import sys

from cpuinfo import cpu
from fcompiler import FCompiler

class SunFCompiler(FCompiler):

    compiler_type = 'sun'
    version_pattern = r'(f90|f95): (Sun|Forte Developer 7) Fortran 95 (?P<version>[^\s]+).*'

    executables = {
        'version_cmd'  : ["f90", "-V"],
        'compiler_f77' : ["f90", "-f77", "-ftrap=%none"],
        'compiler_fix' : ["f90", "-fixed"],
        'compiler_f90' : ["f90"],
        'linker_so'    : ["f90","-Bdynamic","-G"],
        'archiver'     : ["ar", "-cr"],
        'ranlib'       : ["ranlib"]
        }
    module_dir_switch = '-moddir='
    module_include_switch = '-M'
    pic_flags = ['-xcode=pic32']

    def get_opt(self):
        return ['-fast','-dalign']
    def get_arch(self):
        return ['-xtarget=generic']
    def get_libraries(self):
        opt = []
        opt.extend(['fsu','sunmath','mvec','f77compat'])
        return opt

if __name__ == '__main__':
    from scipy_distutils import log
    log.set_verbosity(2)
    from fcompiler import new_fcompiler
    compiler = new_fcompiler(compiler='sun')
    compiler.customize()
    print compiler.get_version()
