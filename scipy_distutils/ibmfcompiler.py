import os
import re
import sys

from fcompiler import FCompiler
import log

class IbmFCompiler(FCompiler):

    compiler_type = 'ibm'
    version_pattern =  r'xlf\(1\)\s*IBM XL Fortran (Advanced Edition |)Version (?P<version>[^\s*]*)'

    executables = {
        'version_cmd'  : ["xlf"],
        'compiler_f77' : ["xlf"],
        'compiler_fix' : ["xlf90", "-qfixed"],
        'compiler_f90' : ["xlf90"],
        'linker_so'    : ["xlf95"],
        'archiver'     : ["ar", "-cr"],
        'ranlib'       : ["ranlib"]
        }

    def get_version(self,*args,**kwds):
        version = FCompiler.get_version(self,*args,**kwds)
        xlf_dir = '/etc/opt/ibmcmp/xlf'
        if version is None and os.path.isdir(xlf_dir):
            # If the output of xlf does not contain version info
            # (that's the case with xlf 8.1, for instance) then
            # let's try another method:
            l = os.listdir(xlf_dir)
            l.sort()
            l.reverse()
            l = [d for d in l if os.path.isfile(os.path.join(xlf_dir,d,'xlf.cfg'))]
            if not l:
                from distutils.version import LooseVersion
                self.version = version = LooseVersion(l[0])
        return version

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
    from fcompiler import new_fcompiler
    compiler = new_fcompiler(compiler='ibm')
    compiler.customize()
    print compiler.get_version()
