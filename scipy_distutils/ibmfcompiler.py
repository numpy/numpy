import os
import sys

from fcompiler import FCompiler
import log

class IbmFCompiler(FCompiler):

    compiler_type = 'ibm'
    version_pattern =  r'xlf\(1\)\s*IBM XL Fortran Version (?P<version>[^\s*]*)'

    executables = {
        'version_cmd'  : ["f77"],
        'compiler_f77' : ["f77"],
        'compiler_fix' : ["xlf", "-qfixed"],
        'compiler_f90' : ["xlf"],
        'linker_so'    : ["xlf"],
        'archiver'     : ["ar", "-cr"],
        'ranlib'       : ["ranlib"]
        }

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
        version = self.get_version(ok_status=[0,10240])
        if version is not None:
            import tempfile
            xlf_cfg = '/etc/opt/ibmcmp/xlf/%s/xlf.cfg' % version
            new_cfg = tempfile.mktemp()+'_xlf.cfg'
            log.info('Creating '+new_cfg)
            fi = open(xlf_cfg,'r')
            fo = open(new_cfg,'w')
            for line in fi.readlines():
                #XXX: Is replacing all occurrences of crt1.o the right
                #     thing to do? What is the pattern of the relevant
                #     line?
                fo.write(line.replace('/crt1.o','/bundle1.o'))
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
