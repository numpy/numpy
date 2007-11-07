#! /usr/bin/env python
# Last Change: Mon Nov 05 06:00 PM 2007 J
import sys
import distutils.sysconfig

# This is a copy of scons/Tools/__init__.py, because scons does not offer any
# public api for this
def tool_list(platform):
    """platform should be the value returned by enbv['PLATFORM'], not
    sys.platform !!!!."""
    # Here, we set the list of default tools as used by numpy.distutils.scons. This
    # is ripped of scons, because scons does not provide a way to get separate
    # default lists for separate tools (e.g linker, C compiler, etc...)

    if str(platform) == 'win32':
        "prefer Microsoft tools on Windows"
        linkers = ['mslink', 'gnulink', 'ilink', 'linkloc', 'ilink32' ]
        c_compilers = ['msvc', 'mingw', 'gcc', 'intelc', 'icl', 'icc', 'cc', 'bcc32' ]
        cxx_compilers = ['msvc', 'intelc', 'icc', 'g++', 'c++', 'bcc32' ]
        assemblers = ['masm', 'nasm', 'gas', '386asm' ]
        fortran_compilers = ['g77', 'ifl', 'cvf', 'f95', 'f90', 'fortran']
        ars = ['mslib', 'ar', 'tlib']
    elif str(platform) == 'os2':
        "prefer IBM tools on OS/2"
        linkers = ['ilink', 'gnulink', 'mslink']
        c_compilers = ['icc', 'gcc', 'msvc', 'cc']
        cxx_compilers = ['icc', 'g++', 'msvc', 'c++']
        assemblers = ['nasm', 'masm', 'gas']
        fortran_compilers = ['ifl', 'g77']
        ars = ['ar', 'mslib']
    elif str(platform) == 'irix':
        "prefer MIPSPro on IRIX"
        linkers = ['sgilink', 'gnulink']
        c_compilers = ['sgicc', 'gcc', 'cc']
        cxx_compilers = ['sgic++', 'g++', 'c++']
        assemblers = ['as', 'gas']
        fortran_compilers = ['f95', 'f90', 'f77', 'g77', 'fortran']
        ars = ['sgiar']
    elif str(platform) == 'sunos':
        "prefer Forte tools on SunOS"
        linkers = ['sunlink', 'gnulink']
        c_compilers = ['suncc', 'gcc', 'cc']
        cxx_compilers = ['sunc++', 'g++', 'c++']
        assemblers = ['as', 'gas']
        fortran_compilers = ['f95', 'f90', 'f77', 'g77', 'fortran']
        ars = ['sunar']
    elif str(platform) == 'hpux':
        "prefer aCC tools on HP-UX"
        linkers = ['hplink', 'gnulink']
        c_compilers = ['hpcc', 'gcc', 'cc']
        cxx_compilers = ['hpc++', 'g++', 'c++']
        assemblers = ['as', 'gas']
        fortran_compilers = ['f95', 'f90', 'f77', 'g77', 'fortran']
        ars = ['ar']
    elif str(platform) == 'aix':
        "prefer AIX Visual Age tools on AIX"
        linkers = ['aixlink', 'gnulink']
        c_compilers = ['aixcc', 'gcc', 'cc']
        cxx_compilers = ['aixc++', 'g++', 'c++']
        assemblers = ['as', 'gas']
        fortran_compilers = ['f95', 'f90', 'aixf77', 'g77', 'fortran']
        ars = ['ar']
    elif str(platform) == 'darwin':
        "prefer GNU tools on Mac OS X, except for some linkers and IBM tools"
        linkers = ['applelink', 'gnulink']
        c_compilers = ['gcc', 'cc']
        cxx_compilers = ['g++', 'c++']
        assemblers = ['as']
        fortran_compilers = ['f95', 'f90', 'g77']
        ars = ['ar']
    else:
        "prefer GNU tools on all other platforms"
        linkers = ['gnulink', 'mslink', 'ilink']
        c_compilers = ['gcc', 'msvc', 'intelc', 'icc', 'cc']
        cxx_compilers = ['g++', 'msvc', 'intelc', 'icc', 'c++']
        assemblers = ['gas', 'nasm', 'masm']
        fortran_compilers = ['f95', 'f90', 'g77', 'ifort', 'ifl', 'fortran']
        ars = ['ar', 'mslib']

    other_tools = ['BitKeeper', 'CVS', 'dmd', 'dvipdf', 'dvips', 'gs', 'jar',
            'javac', 'javah', 'latex', 'lex', 'm4', 'midl', 'msvs', 'pdflatex',
            'pdftex', 'Perforce', 'RCS', 'rmic', 'rpcgen', 'SCCS', 'swig',
            'tar', 'tex', 'yacc', 'zip']
    return linkers, c_compilers, cxx_compilers, assemblers, fortran_compilers, \
           ars, other_tools

# Handling compiler configuration: only flags which change how to build object
# files. Nothing related to linking, search path, etc... should be given here.
# Basically, limit yourself to optimization/debug/warning flags.

# XXX: customization from site.cfg or other ?
class CompilerConfig:
    def __init__(self, optim = None, warn = None, debug = None, debug_symbol =
                 None, thread = None, extra = None):
        # XXX: several level of optimizations ?
        self.optim = optim
        # XXX: several level of warnings ?
        self.warn = warn
        # To enable putting debugging info in binaries
        self.debug_symbol = debug_symbol
        # To enable friendly debugging
        self.debug = debug
        # XXX
        self.thread = thread
        # XXX
        self.extra = extra

    def get_flags_dict(self):
        d = {'NUMPY_OPTIM_CFLAGS' : self.optim,
                'NUMPY_WARN_CFLAGS' : self.warn,
                'NUMPY_THREAD_CFLAGS' : self.thread,
                'NUMPY_EXTRA_CFLAGS' : self.debug,
                'NUMPY_DEBUG_CFLAGS' : self.debug,
                'NUMPY_DEBUG_SYMBOL_CFLAGS' : self.debug_symbol}
        for k, v in d.items():
            if v is None:
                d[k] = []
        return d

# It seems that scons consider any option with space in it as a multi option,
# which breaks command line options. So just don't put space.
def get_cc_config(name):
    if name == 'gcc':
        if distutils.sysconfig.get_config_vars('LDFLAGS')[0].find('-pthread'):
            thread = ['-pthread']
        else:
            thread = []
        cfg = CompilerConfig(optim = ['-O2', '-fno-strict-aliasing', '-DNDEBUG'],
                             warn = ['-Wall', '-Wstrict-prototypes'],
                             debug_symbol = ['-g'], 
                             thread = thread)
    elif name == 'intelc':
        if sys.platform[:5] == 'win32':
            raise NotImplementedError('FIXME: intel compiler on windows not '\
                                      ' supported yet')
            
        cfg = CompilerConfig(optim = ['-O2', '-fno-strict-aliasing', '-DNDEBUG'],
                             warn = ['-Wall', '-Wstrict-prototypes'],
                             debug_symbol = ['-g'],
                             thread = ['-pthread'])
    elif name == 'msvc':
        # XXX: distutils part of customization:
        # if self.__arch == "Intel":
        #     self.compile_options = [ '/nologo', '/Ox', '/MD', '/W3', '/GX' , '/DNDEBUG']
        #     self.compile_options_debug = ['/nologo', '/Od', '/MDd', '/W3', '/GX', '/Z7', '/D_DEBUG']
        # else:
        #     # Win64
        #     self.compile_options = [ '/nologo', '/Ox', '/MD', '/W3', '/GS-' ,
        #     '/DNDEBUG']
        #     self.compile_options_debug = ['/nologo', '/Od', '/MDd', '/W3', '/GS-',
        #     '/Z7', '/D_DEBUG']
        # 
        # self.ldflags_shared = ['/DLL', '/nologo', '/INCREMENTAL:NO']
        # if self.__version >= 7:
        #     self.ldflags_shared_debug = [
        #     '/DLL', '/nologo', '/INCREMENTAL:no', '/DEBUG'
        #     ]
        # else:
        #     self.ldflags_shared_debug = [
        #     '/DLL', '/nologo', '/INCREMENTAL:no', '/pdb:None', '/DEBUG'
        #             ]
        # self.ldflags_static = [ '/nologo']

        cfg = CompilerConfig(optim = ['/Ox', '/DNDEBUG'],
                             warn = ['/W3', '/Wall'],
                             thread = ['/MD', '/GX'], 
                             extra = ['/nologo'])
    else:
        # For not yet supported compiler, just put everything in optims from
        # distutils
        cfg = CompilerConfig(optim =
                distutils.sysconfig.get_config_vars('CFLAGS'))

    return cfg
