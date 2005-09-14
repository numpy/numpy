import os
import sys

from cpuinfo import cpu
from fcompiler import FCompiler

class HPUXFCompiler(FCompiler):

    compiler_type = 'hpux'
    version_pattern =  r'HP F90 (?P<version>[^\s*,]*)'

    executables = {
        'version_cmd'  : ["f90", "+version"],
        'compiler_f77' : ["f90"],
        'compiler_fix' : ["f90"],
        'compiler_f90' : ["f90"],
        'linker_so'    : None,
        'archiver'     : ["ar", "-cr"],
        'ranlib'       : ["ranlib"]
        }
    module_dir_switch = None #XXX: fix me
    module_include_switch = None #XXX: fix me
    pic_flags = ['+pic=long']
    def get_flags(self):
        return self.pic_flags + ['+ppu']
    def get_flags_opt(self):
        return ['-O3']
    def get_libraries(self):
        return ['m']
    def get_version(self, force=0, ok_status=[256,0]):
        # XXX status==256 may indicate 'unrecognized option' or
        #     'no input file'. So, version_cmd needs more work.
        return FCompiler.get_version(self,force,ok_status)

if __name__ == '__main__':
    from distutils import log
    log.set_verbosity(10)
    from fcompiler import new_fcompiler
    compiler = new_fcompiler(compiler='hpux')
    compiler.customize()
    print compiler.get_version()
