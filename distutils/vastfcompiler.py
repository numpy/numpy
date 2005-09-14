import os
import sys

from cpuinfo import cpu
from gnufcompiler import GnuFCompiler
#from fcompiler import FCompiler

class VastFCompiler(GnuFCompiler):

    compiler_type = 'vast'
    version_pattern = r'\s*Pacific-Sierra Research vf90 '\
                      '(Personal|Professional)\s+(?P<version>[^\s]*)'

    # VAST f90 does not support -o with -c. So, object files are created
    # to the current directory and then moved to build directory
    object_switch = ' && function _mvfile { mv -v `basename $1` $1 ; } && _mvfile '

    executables = {
        'version_cmd'  : ["vf90", "-v"],
        'compiler_f77' : ["g77"],
        'compiler_fix' : ["f90", "-Wv,-ya"],
        'compiler_f90' : ["f90"],
        'linker_so'    : ["f90"],
        'archiver'     : ["ar", "-cr"],
        'ranlib'       : ["ranlib"]
        }
    module_dir_switch = None  #XXX Fix me
    module_include_switch = None #XXX Fix me

    def get_version_cmd(self):
        f90 = self.compiler_f90[0]
        d,b = os.path.split(f90)
        vf90 = os.path.join(d,'v'+b)
        return vf90

    def get_flags_arch(self):
        vast_version = self.get_version()
        gnu = GnuFCompiler()
        gnu.customize()
        self.version = gnu.get_version()
        opt = GnuFCompiler.get_flags_arch(self)
        self.version = vast_version
        return opt

if __name__ == '__main__':
    from distutils import log
    log.set_verbosity(2)
    from fcompiler import new_fcompiler
    compiler = new_fcompiler(compiler='vast')
    compiler.customize()
    print compiler.get_version()
