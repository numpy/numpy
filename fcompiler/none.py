
from numpy.distutils.fcompiler import FCompiler

class NoneFCompiler(FCompiler):

    compiler_type = 'none'

    executables = {'compiler_f77':['/path/to/nowhere/none'],
                   'compiler_f90':['/path/to/nowhere/none'],
                   'compiler_fix':['/path/to/nowhere/none'],
                   'linker_so':['/path/to/nowhere/none'],
                   'archiver':['/path/to/nowhere/none'],
                   'ranlib':['/path/to/nowhere/none'],
                   'version_cmd':['/path/to/nowhere/none'],
                   }


if __name__ == '__main__':
    from distutils import log
    log.set_verbosity(2)
    from numpy.distutils.fcompiler import new_fcompiler
    compiler = NoneFCompiler()
    compiler.customize()
    print compiler.get_version()
