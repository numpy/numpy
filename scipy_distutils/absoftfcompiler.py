
# http://www.absoft.com/literature/osxuserguide.pdf
# http://www.absoft.com/documentation.html

import os
import sys

from cpuinfo import cpu
from fcompiler import FCompiler, dummy_fortran_file
from misc_util import cyg2win32

class AbsoftFCompiler(FCompiler):

    compiler_type = 'absoft'
    #version_pattern = r'FORTRAN 77 Compiler (?P<version>[^\s*,]*).*?Absoft Corp'
    version_pattern = r'(f90:.*?Absoft Pro FORTRAN Version|FORTRAN 77 Compiler)'+\
                      r' (?P<version>[^\s*,]*)(.*?Absoft Corp|)'

    # samt5735(8)$ f90 -V -c dummy.f
    # f90: Copyright Absoft Corporation 1994-2002; Absoft Pro FORTRAN Version 8.0
    # Note that fink installs g77 as f77, so need to use f90 for detection.

    executables = {
        'version_cmd'  : ["f90", "-V -c %(fname)s.f -o %(fname)s.o" \
                          % {'fname':cyg2win32(dummy_fortran_file())}],
        'compiler_f77' : ["f77"],
        'compiler_fix' : ["f90"],
        'compiler_f90' : ["f90"],
        'linker_so'    : ["f90"],
        'archiver'     : ["ar", "-cr"],
        'ranlib'       : ["ranlib"]
        }

    if os.name=='nt':
        library_switch = '/out:'      #No space after /out:!

    module_dir_switch = None
    module_include_switch = '-p'

    def get_flags_linker_so(self):
        if os.name=='nt':
            opt = ['/dll']
        else:
            opt = ["-K","shared"]
        return opt

    def library_dir_option(self, dir):
        if os.name=='nt':
            return ['-link','/PATH:"%s"' % (dir)]
        return "-L" + dir

    def library_option(self, lib):
        if os.name=='nt':
            return '%s.lib' % (lib)
        return "-l" + lib

    def get_library_dirs(self):
        opt = FCompiler.get_library_dirs(self)
        d = os.environ.get('ABSOFT')
        if d:
            opt.append(os.path.join(d,'LIB'))
        return opt

    def get_libraries(self):
        opt = FCompiler.get_libraries(self)
        opt.extend(['fio','f90math','fmath'])
        if os.name =='nt':
            opt.append('COMDLG32')
        return opt

    def get_flags(self):
        opt = FCompiler.get_flags(self)
        if os.name != 'nt':
            opt.extend(['-s'])
            if self.get_version():
                if self.get_version()>='8.2':
                    opt.append('-fpic')
        return opt

    def get_flags_f77(self):
        opt = FCompiler.get_flags_f77(self)
        opt.extend(['-N22','-N90','-N110'])
        v = self.get_version()
        if os.name == 'nt':
            if v and v>='8.0':
                opt.extend(['-f','-N15'])
        else:
            opt.append('-f')
            if v:
                if v<='4.6':
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
        if self.get_version():
            if self.get_version()>'4.6':
                opt.extend(["-YDEALLOC=ALL"])                
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
