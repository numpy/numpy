"""
unixccompiler - can handle very long argument lists for ar.
"""

import os
from distutils import unixccompiler

import log

class UnixCCompiler(unixccompiler.UnixCCompiler):

    def create_static_lib(self, objects, output_libname,
                          output_dir=None, debug=0, target_lang=None):
        objects, output_dir = self._fix_object_args(objects, output_dir)
        
        output_filename = \
            self.library_filename(output_libname, output_dir=output_dir)
        
        if self._need_link(objects, output_filename):
            self.mkpath(os.path.dirname(output_filename))
            tmp_objects = objects + self.objects
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
                try:
                    self.spawn(self.ranlib + [output_filename])
                except DistutilsExecError, msg:
                    raise LibError, msg
        else:
            log.debug("skipping %s (up-to-date)", output_filename)
