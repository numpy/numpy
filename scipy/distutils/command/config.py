# Added Fortran compiler support to config. Currently useful only for
# try_compile call. try_run works but is untested for most of Fortran
# compilers (they must define linker_exe first).
# Pearu Peterson

from distutils.command.config import config as old_config
from distutils.command.config import LANG_EXT
LANG_EXT['f77'] = '.f'
LANG_EXT['f90'] = '.f90'

class config(old_config):
    old_config.user_options += [
        ('fcompiler=', None,
         "specify the Fortran compiler type"),
        ]

    def initialize_options(self):
        self.fcompiler = None
        old_config.initialize_options(self)
        return

    def finalize_options(self):
        old_config.finalize_options(self)
        f = self.distribution.get_command_obj('config_fc')
        self.set_undefined_options('config_fc',
                                   ('fcompiler', 'fcompiler'))
        return

    def _check_compiler (self):
        old_config._check_compiler(self)
        from scipy.distutils.fcompiler import FCompiler, new_fcompiler
        if not isinstance(self.fcompiler, FCompiler):
            self.fcompiler = new_fcompiler(compiler=self.fcompiler,
                                           dry_run=self.dry_run, force=1)
            self.fcompiler.customize(self.distribution)
            self.fcompiler.customize_cmd(self)
            self.fcompiler.show_customization()
        return

    def _wrap_method(self,mth,lang,args):
        from distutils.ccompiler import CompileError
        from distutils.errors import DistutilsExecError
        save_compiler = self.compiler
        if lang in ['f77','f90']:
            self.compiler = self.fcompiler
        try:
            ret = mth(*((self,)+args))
        except (DistutilsExecError,CompileError),msg:
            self.compiler = save_compiler
            raise CompileError
        self.compiler = save_compiler
        return ret

    def _compile (self, body, headers, include_dirs, lang):
        return self._wrap_method(old_config._compile,lang,
                                 (body, headers, include_dirs, lang))

    def _link (self, body,
               headers, include_dirs,
               libraries, library_dirs, lang):
        return self._wrap_method(old_config._link,lang,
                                 (body, headers, include_dirs,
                                  libraries, library_dirs, lang))
