
# http://www.absoft.com/literature/osxuserguide.pdf
# http://www.absoft.com/documentation.html

import os
import sys

from cpuinfo import cpu
from fcompiler import FCompiler, dummy_fortran_file

class AbsoftFCompiler(FCompiler):

    compiler_type = 'absoft'
    version_pattern = r'FORTRAN 77 Compiler (?P<version>[^\s*,]*).*?Absoft Corp'

    executables = {
        'version_cmd'  : ["f77", "-V -c %(fname)s.f -o %(fname)s.o" \
                          % {'fname':dummy_fortran_file()}],
        'compiler_f77' : ["f77"],
        'compiler_fix' : ["f90"],
        'compiler_f90' : ["f90"],
        'linker_so'    : ["f77","-shared"],
        'archiver'     : ["ar", "-cr"],
        'ranlib'       : ["ranlib"]
        }

    if os.name != 'nt':
        pic_flags = ['-fpic']
    module_dir_switch = None
    module_include_switch = '-p '

    def get_library_dirs(self):
        opt = FCompiler.get_library_dirs(self)
        d = os.environ.get('ABSOFT')
        if d:
            opt.append(d)
        return opt

    def get_libraries(self):
        opt = FCompiler.get_libraries(self)
        opt.extend(['fio','f77math','f90math'])
        if os.name =='nt':
            opt.append('COMDLG32')
        return opt

    def get_flags(self):
        opt = FCompiler.get_flags(self)
        if os.name != 'nt':
            opt.extend(['-s'])
        return opt

    def get_flags_f77(self):
        opt = FCompiler.get_flags_f77(self)
        opt.extend(['-N22','-N90','-N110'])
        if os.name != 'nt':
            opt.append('-f')
            if self.get_version():
                if self.get_version()<='4.6':
                    opt.append('-B108')
                else:
                    # Though -N15 is undocumented, it works with
                    # Absoft 8.0 on Linux
                    opt.append('-N15')
        return opt

    def get_flags_f90(self):
        opt = FCompiler.get_flags_f90(self)
        opt.extend(["-YCFRL=1","-YCOM_NAMES=LCS","-YCOM_PFX","-YEXT_PFX",
                    "-YCOM_SFX=_","-YEXT_SFX=_","-YEXT_NAMES=LCS"])
        return opt

    def get_flags_fix(self):
        opt = FCompiler.get_flags_fix(self)
        opt.extend(["-YCFRL=1","-YCOM_NAMES=LCS","-YCOM_PFX","-YEXT_PFX",
                    "-YCOM_SFX=_","-YEXT_SFX=_","-YEXT_NAMES=LCS"])
        opt.extend(["-f","fixed"])
        return opt

    def get_flags_opt(self):
        opt = ['-O']
        return opt

if __name__ == '__main__':
    from distutils import log
    log.set_verbosity(2)
    from fcompiler import new_fcompiler
    compiler = new_fcompiler(compiler='absoft')
    compiler.customize()
    print compiler.get_version()
