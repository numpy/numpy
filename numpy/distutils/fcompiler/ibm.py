import os
import re
import sys

from numpy.distutils.fcompiler import FCompiler
from numpy.distutils.exec_command import exec_command, find_executable
from distutils import log
from distutils.sysconfig import get_python_lib

class IbmFCompiler(FCompiler):

    compiler_type = 'ibm'
    version_pattern =  r'(xlf\(1\)\s*|)IBM XL Fortran ((Advanced Edition |)Version |Enterprise Edition V)(?P<version>[^\s*]*)'
    #IBM XL Fortran Enterprise Edition V10.1 for AIX \nVersion: 10.01.0000.0004
    executables = {
        'version_cmd'  : ["xlf","-qversion"],
        'compiler_f77' : ["xlf"],
        'compiler_fix' : ["xlf90", "-qfixed"],
        'compiler_f90' : ["xlf90"],
        'linker_so'    : ["xlf95"],
        'archiver'     : ["ar", "-cr"],
        'ranlib'       : ["ranlib"]
        }

    def get_version(self,*args,**kwds):
        version = FCompiler.get_version(self,*args,**kwds)

        if version is None and sys.platform.startswith('aix'):
            # use lslpp to find out xlf version
            lslpp = find_executable('lslpp')
            xlf = find_executable('xlf')
            if os.path.exists(xlf) and os.path.exists(lslpp):
                s,o = exec_command(lslpp + ' -Lc xlfcmp')
                m = re.search('xlfcmp:(?P<version>\d+([.]\d+)+)', o)
                if m: version = m.group('version')

        xlf_dir = '/etc/opt/ibmcmp/xlf'
        if version is None and os.path.isdir(xlf_dir):
            # linux:
            # If the output of xlf does not contain version info
            # (that's the case with xlf 8.1, for instance) then
            # let's try another method:
            l = os.listdir(xlf_dir)
            l.sort()
            l.reverse()
            l = [d for d in l if os.path.isfile(os.path.join(xlf_dir,d,'xlf.cfg'))]
            if l:
                from distutils.version import LooseVersion
                self.version = version = LooseVersion(l[0])
        return version

    def get_linker_so(self):
        if sys.platform.startswith('aix'):
            python_lib = get_python_lib(standard_lib=1)
            ld_so_aix = os.path.join(python_lib, 'config', 'ld_so_aix')
            python_exp = os.path.join(python_lib, 'config', 'python.exp')
            return [ld_so_aix, self.executables['linker_so'], python_exp]
        else:
            return FCompiler.get_linker_so(self)

    def get_flags(self):
        return ['-qextname']

    def get_flags_debug(self):
        return ['-g']

    def get_flags_linker_so(self):
        opt = []
        if sys.platform=='darwin':
            opt.append('-Wl,-bundle,-flat_namespace,-undefined,suppress')
        else:
            opt.append('-bshared')
        version = self.get_version(ok_status=[0,40])
        if version is not None:
            import tempfile
            if sys.platform.startswith('aix'):
                xlf_cfg = '/etc/xlf.cfg'
            else:
                xlf_cfg = '/etc/opt/ibmcmp/xlf/%s/xlf.cfg' % version
            new_cfg = tempfile.mktemp()+'_xlf.cfg'
            log.info('Creating '+new_cfg)
            fi = open(xlf_cfg,'r')
            fo = open(new_cfg,'w')
            crt1_match = re.compile(r'\s*crt\s*[=]\s*(?P<path>.*)/crt1.o').match
            for line in fi.readlines():
                m = crt1_match(line)
                if m:
                    fo.write('crt = %s/bundle1.o\n' % (m.group('path')))
                else:
                    fo.write(line)
            fi.close()
            fo.close()
            opt.append('-F'+new_cfg)
        return opt

    def get_flags_opt(self):
        return ['-O5']

if __name__ == '__main__':
    from distutils import log
    log.set_verbosity(2)
    from numpy.distutils.fcompiler import new_fcompiler
    #compiler = new_fcompiler(compiler='ibm')
    compiler = IbmFCompiler()
    compiler.customize()
    print compiler.get_version()
