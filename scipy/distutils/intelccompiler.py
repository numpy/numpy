
import os
from distutils.unixccompiler import UnixCCompiler
from scipy.distutils.exec_command import find_executable

class IntelCCompiler(UnixCCompiler):

    """ A modified Intel compiler compatible with an gcc built Python.
    """

    compiler_type = 'intel'
    cc_exe = 'icc'

    def __init__ (self, verbose=0, dry_run=0, force=0):
        UnixCCompiler.__init__ (self, verbose,dry_run, force)
        self.linker = self.cc_exe
        self.set_executable(linker_so=self.linker + ' -shared')

class IntelItaniumCompiler(IntelCCompiler):
    compiler_type = 'intele'

    for cc_exe in map(find_executable,['ecc','icc']):
        if os.path.isfile(cc_exe):
            break
