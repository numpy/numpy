# Added Fortran compiler support to config. Currently useful only for
# try_compile call. try_run works but is untested for most of Fortran
# compilers (they must define linker_exe first).
# Pearu Peterson

import os, signal
from distutils.command.config import config as old_config
from distutils.command.config import LANG_EXT
from distutils import log
from distutils.file_util import copy_file
from numpy.distutils.exec_command import exec_command

LANG_EXT['f77'] = '.f'
LANG_EXT['f90'] = '.f90'

class config(old_config):
    old_config.user_options += [
        ('fcompiler=', None, "specify the Fortran compiler type"),
        ]

    def initialize_options(self):
        self.fcompiler = None
        old_config.initialize_options(self)

    def _check_compiler (self):
        old_config._check_compiler(self)
        from numpy.distutils.fcompiler import FCompiler, new_fcompiler
        if not isinstance(self.fcompiler, FCompiler):
            self.fcompiler = new_fcompiler(compiler=self.fcompiler,
                                           dry_run=self.dry_run, force=1)
            if self.fcompiler is not None and self.fcompiler.get_version():
                self.fcompiler.customize(self.distribution)
                self.fcompiler.customize_cmd(self)
                self.fcompiler.show_customization()

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
        if self.compiler.compiler_type=='msvc':
            libraries = (libraries or [])[:]
            library_dirs = (library_dirs or [])[:]
            if lang in ['f77','f90']:
                lang = 'c' # always use system linker when using MSVC compiler
                if self.fcompiler:
                    for d in self.fcompiler.library_dirs or []:
                        # correct path when compiling in Cygwin but with
                        # normal Win Python
                        if d.startswith('/usr/lib'):
                            s,o = exec_command(['cygpath', '-w', d],
                                               use_tee=False)
                            if not s: d = o
                        library_dirs.append(d)
                    for libname in self.fcompiler.libraries or []:
                        if libname not in libraries:
                            libraries.append(libname)
            for libname in libraries:
                if libname.startswith('msvc'): continue
                fileexists = False
                for libdir in library_dirs or []:
                    libfile = os.path.join(libdir,'%s.lib' % (libname))
                    if os.path.isfile(libfile):
                        fileexists = True
                        break
                if fileexists: continue
                # make g77-compiled static libs available to MSVC
                fileexists = False
                for libdir in library_dirs:
                    libfile = os.path.join(libdir,'lib%s.a' % (libname))
                    if os.path.isfile(libfile):
                        # copy libname.a file to name.lib so that MSVC linker
                        # can find it
                        libfile2 = os.path.join(libdir,'%s.lib' % (libname))
                        copy_file(libfile, libfile2)
                        self.temp_files.append(libfile2)
                        fileexists = True
                        break
                if fileexists: continue
                log.warn('could not find library %r in directories %s' \
                         % (libname, library_dirs))
        return self._wrap_method(old_config._link,lang,
                                 (body, headers, include_dirs,
                                  libraries, library_dirs, lang))

    def check_func(self, func,
                   headers=None, include_dirs=None,
                   libraries=None, library_dirs=None,
                   decl=False, call=False, call_args=None):
        # clean up distutils's config a bit: add void to main(), and
        # return a value.
        self._check_compiler()
        body = []
        if decl:
            body.append("int %s ();" % func)
        body.append("int main (void) {")
        if call:
            if call_args is None:
                call_args = ''
            body.append("  %s(%s);" % (func, call_args))
        else:
            body.append("  %s;" % func)
        body.append("  return 0;")
        body.append("}")
        body = '\n'.join(body) + "\n"

        return self.try_link(body, headers, include_dirs,
                             libraries, library_dirs)

    def get_output(self, body, headers=None, include_dirs=None,
                   libraries=None, library_dirs=None,
                   lang="c"):
        """Try to compile, link to an executable, and run a program
        built from 'body' and 'headers'. Returns the exit status code
        of the program and its output.
        """
        from distutils.ccompiler import CompileError, LinkError
        self._check_compiler()
        exitcode, output = 255, ''
        try:
            src, obj, exe = self._link(body, headers, include_dirs,
                                       libraries, library_dirs, lang)
            exe = os.path.join('.', exe)
            exitstatus, output = exec_command(exe, execute_in='.')
            if hasattr(os, 'WEXITSTATUS'):
                exitcode = os.WEXITSTATUS(exitstatus)
                if os.WIFSIGNALED(exitstatus):
                    sig = os.WTERMSIG(exitstatus)
                    log.error('subprocess exited with signal %d' % (sig,))
                    if sig == signal.SIGINT:
                        # control-C
                        raise KeyboardInterrupt
            else:
                exitcode = exitstatus
            log.info("success!")
        except (CompileError, LinkError):
            log.info("failure.")

        self._clean()
        return exitcode, output

