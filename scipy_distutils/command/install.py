from types import StringType
from distutils.command.install import *
from distutils.command.install import install as old_install
from distutils.util import convert_path
from distutils.file_util import write_file
from distutils.errors import DistutilsOptionError

#install support for Numeric.pth setup
class install(old_install):
    def finalize_options (self):
        old_install.finalize_options(self)
        self.install_lib = self.install_libbase
        
    def handle_extra_path (self):
        if self.extra_path is None:
            self.extra_path = self.distribution.extra_path

        if self.extra_path is not None:
            if type(self.extra_path) is StringType:
                self.extra_path = string.split(self.extra_path, ',')
            if len(self.extra_path) == 1:
                path_file = extra_dirs = self.extra_path[0]
            elif len(self.extra_path) == 2:
                (path_file, extra_dirs) = self.extra_path
            else:
                raise DistutilsOptionError, \
                      "'extra_path' option must be a list, tuple, or " + \
                      "comma-separated string with 1 or 2 elements"

            # convert to local form in case Unix notation used (as it
            # should be in setup scripts)
            extra_dirs = convert_path(extra_dirs)

        else:
            path_file = None
            extra_dirs = ''

        # XXX should we warn if path_file and not extra_dirs? (in which
        # case the path file would be harmless but pointless)
        self.path_file = path_file
        self.extra_dirs = ''
        self.pth_file = extra_dirs

    # handle_extra_path ()

    def create_path_file (self):
        filename = os.path.join(self.install_libbase,
                                self.path_file + ".pth")
        if self.install_path_file:
            self.execute(write_file,
                         (filename, [self.pth_file]),
                         "creating %s" % filename)
        else:
            self.warn("path file '%s' not created" % filename)