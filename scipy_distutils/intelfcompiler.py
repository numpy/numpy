# http://developer.intel.com/software/products/compilers/flin/

import os
import sys

from cpuinfo import cpu
from fcompiler import FCompiler, dummy_fortran_file
from exec_command import find_executable

class IntelFCompiler(FCompiler):

    compiler_type = 'intel'
    version_pattern = r'Intel\(R\) Fortran Compiler for 32-bit '\
                      'applications, Version (?P<version>[^\s*]*)'

    for fc_exe in map(find_executable,['ifort','ifc']):
        if os.path.isfile(fc_exe):
            break

    executables = {
        'version_cmd'  : [fc_exe, "-FI -V -c %(fname)s.f -o %(fname)s.o" \
                          % {'fname':dummy_fortran_file()}],
        'compiler_f77' : [fc_exe,"-72","-w90","-w95"],
        'compiler_fix' : [fc_exe,"-FI"],
        'compiler_f90' : [fc_exe],
        'linker_so'    : [fc_exe,"-shared"],
        'archiver'     : ["ar", "-cr"],
        'ranlib'       : ["ranlib"]
        }

    pic_flags = ['-KPIC']
    module_dir_switch = '-module ' # Don't remove ending space!
    module_include_switch = '-I'

    def get_flags(self):
        opt = self.pic_flags + ["-cm"]
        return opt

    def get_flags_free(self):
        return ["-FR"]

    def get_flags_opt(self):
        return ['-O3','-unroll']

    def get_flags_arch(self):
        opt = []
        if cpu.has_fdiv_bug():
            opt.append('-fdiv_check')
        if cpu.has_f00f_bug():
            opt.append('-0f_check')
        if cpu.is_PentiumPro() or cpu.is_PentiumII():
            opt.extend(['-tpp6','-xi'])
        elif cpu.is_PentiumIII():
            opt.append('-tpp6')
        elif cpu.is_Pentium():
            opt.append('-tpp5')
        elif cpu.is_PentiumIV() or cpu.is_XEON():
            opt.extend(['-tpp7','-xW'])
        if cpu.has_mmx():
            opt.append('-xM')
        return opt

    def get_flags_linker_so(self):
        opt = FCompiler.get_flags_linker_so(self)
        v = self.get_version()
        if v and v >= '8.0':
            opt.append('-nofor_main')
        return opt

class IntelItaniumFCompiler(IntelFCompiler):
    compiler_type = 'intele'
    version_pattern = r'Intel\(R\) Fortran 90 Compiler Itanium\(TM\) Compiler'\
                      ' for the Itanium\(TM\)-based applications,'\
                      ' Version (?P<version>[^\s*]*)'

    for fc_exe in map(find_executable,['efort','efc','ifort']):
        if os.path.isfile(fc_exe):
            break

    executables = {
        'version_cmd'  : [fc_exe, "-FI -V -c %(fname)s.f -o %(fname)s.o" \
                          % {'fname':dummy_fortran_file()}],
        'compiler_f77' : [fc_exe,"-FI","-w90","-w95"],
        'compiler_fix' : [fc_exe,"-FI"],
        'compiler_f90' : [fc_exe],
        'linker_so'    : [fc_exe,"-shared"],
        'archiver'     : ["ar", "-cr"],
        'ranlib'       : ["ranlib"]
        }

class IntelVisualFCompiler(FCompiler):

    compiler_type = 'intelv'
    version_pattern = r'Intel\(R\) Fortran Compiler for 32-bit applications, '\
                      'Version (?P<version>[^\s*]*)'

    ar_exe = 'lib.exe'
    fc_exe = 'ifl'
    if sys.platform=='win32':
        from distutils.msvccompiler import MSVCCompiler
        ar_exe = MSVCCompiler().lib

    executables = {
        'version_cmd'  : [fc_exe, "-FI -V -c %(fname)s.f -o %(fname)s.o" \
                          % {'fname':dummy_fortran_file()}],
        'compiler_f77' : [fc_exe,"-FI","-w90","-w95"],
        'compiler_fix' : [fc_exe,"-FI","-4L72","-w"],
        'compiler_f90' : [fc_exe],
        'linker_so'    : [fc_exe,"-shared"],
        'archiver'     : [ar_exe, "/verbose", "/OUT:"],
        'ranlib'       : None
        }

    compile_switch = '/c '
    object_switch = '/Fo'     #No space after /Fo!
    library_switch = '/OUT:'  #No space after /OUT:!
    module_dir_switch = '/module:' #No space after /module:
    module_include_switch = '/I'

    def get_flags(self):
        opt = ['/nologo','/MD','/nbs','/Qlowercase','/us']
        return opt

    def get_flags_free(self):
        return ["-FR"]

    def get_flags_debug(self):
        return ['/4Yb','/d2']

    def get_flags_opt(self):
        return ['/O3','/Qip','/Qipo','/Qipo_obj']

    def get_flags_arch(self):
        opt = []
        if cpu.is_PentiumPro() or cpu.is_PentiumII():
            opt.extend(['/G6','/Qaxi'])
        elif cpu.is_PentiumIII():
            opt.extend(['/G6','/QaxK'])
        elif cpu.is_Pentium():
            opt.append('/G5')
        elif cpu.is_PentiumIV():
            opt.extend(['/G7','/QaxW'])
        if cpu.has_mmx():
            opt.append('/QaxM')
        return opt

class IntelItaniumVisualFCompiler(IntelVisualFCompiler):

    compiler_type = 'intelev'
    version_pattern = r'Intel\(R\) Fortran 90 Compiler Itanium\(TM\) Compiler'\
                      ' for the Itanium\(TM\)-based applications,'\
                      ' Version (?P<version>[^\s*]*)'

    fc_exe = 'efl' # XXX this is a wild guess
    ar_exe = IntelVisualFCompiler.ar_exe

    executables = {
        'version_cmd'  : [fc_exe, "-FI -V -c %(fname)s.f -o %(fname)s.o" \
                          % {'fname':dummy_fortran_file()}],
        'compiler_f77' : [fc_exe,"-FI","-w90","-w95"],
        'compiler_fix' : [fc_exe,"-FI","-4L72","-w"],
        'compiler_f90' : [fc_exe],
        'linker_so'    : [fc_exe,"-shared"],
        'archiver'     : [ar_exe, "/verbose", "/OUT:"],
        'ranlib'       : None
        }

if __name__ == '__main__':
    from distutils import log
    log.set_verbosity(2)
    from fcompiler import new_fcompiler
    compiler = new_fcompiler(compiler='intel')
    compiler.customize()
    print compiler.get_version()
