"""
unixccompiler - can handle very long argument lists for ar.
"""

import os
from distutils import unixccompiler
from distutils.errors import DistutilsExecError, LinkError
from types import StringType, NoneType

from ccompiler import gen_lib_options, CCompiler
import log
from exec_command import exec_command

class UnixCCompiler(unixccompiler.UnixCCompiler):

    def _compile(self, obj, src, *args):
        log.info('%s: %s' % (os.path.basename(self.compiler_so[0]),src))
        return unixccompiler.UnixCCompiler._compile(self, obj, src, *args)

    def spawn(self, cmd):
        s,o = exec_command(cmd)
        if s:
            if type(cmd) is type([]):
                cmd = ' '.join(cmd)
            raise DistutilsExecError,\
                  'Command "%s" failed with exit status %d' % (cmd, s)

    def create_static_lib(self, objects, output_libname,
                          output_dir=None, debug=0, target_lang=None):
        objects, output_dir = self._fix_object_args(objects, output_dir)
        
        output_filename = \
            self.library_filename(output_libname, output_dir=output_dir)
        
        if self._need_link(objects, output_filename):
            self.mkpath(os.path.dirname(output_filename))
            tmp_objects = objects + self.objects
            log.info('%s:> %s' % (os.path.basename(self.archiver[0]),
                                 output_filename))
            while tmp_objects:
                objects = tmp_objects[:50]
                tmp_objects = tmp_objects[50:]
                self.spawn(self.archiver +
                           [output_filename] +
                           objects)
            
            # Not many Unices required ranlib anymore -- SunOS 4.x is, I
            # think the only major Unix that does.  Maybe we need some
            # platform intelligence here to skip ranlib if it's not
            # needed -- or maybe Python's configure script took care of
            # it for us, hence the check for leading colon.
            if self.ranlib:
                log.info('%s:@ %s' % (os.path.basename(self.ranlib[0]),
                                      output_filename))
                try:
                    self.spawn(self.ranlib + [output_filename])
                except DistutilsExecError, msg:
                    raise LibError, msg
        else:
            log.debug("skipping %s (up-to-date)", output_filename)

    def link(self, target_desc, objects,
             output_filename, output_dir=None, libraries=None,
             library_dirs=None, runtime_library_dirs=None,
             export_symbols=None, debug=0, extra_preargs=None,
             extra_postargs=None, build_temp=None, target_lang=None):
        objects, output_dir = self._fix_object_args(objects, output_dir)
        libraries, library_dirs, runtime_library_dirs = \
            self._fix_lib_args(libraries, library_dirs, runtime_library_dirs)

        lib_opts = gen_lib_options(self, library_dirs, runtime_library_dirs,
                                   libraries)
        if type(output_dir) not in (StringType, NoneType):
            raise TypeError, "'output_dir' must be a string or None"
        if output_dir is not None:
            output_filename = os.path.join(output_dir, output_filename)

        if self._need_link(objects, output_filename):
            ld_args = (objects + self.objects +
                       lib_opts + ['-o', output_filename])
            if debug:
                ld_args[:0] = ['-g']
            if extra_preargs:
                ld_args[:0] = extra_preargs
            if extra_postargs:
                ld_args.extend(extra_postargs)
            self.mkpath(os.path.dirname(output_filename))
            try:
                if target_desc == CCompiler.EXECUTABLE:
                    linker = self.linker_exe[:]
                else:
                    linker = self.linker_so[:]
                if target_lang == "c++" and self.compiler_cxx:
                    linker[0] = self.compiler_cxx[0]
                log.info('%s:> %s' % (os.path.basename(linker[0]),
                                      output_filename))
                self.spawn(linker + ld_args)
            except DistutilsExecError, msg:
                raise LinkError, msg
        else:
            log.debug("skipping %s (up-to-date)", output_filename)
