from distutils.command.sdist import *
from distutils.command.sdist import sdist as old_sdist

class sdist(old_sdist):
    def add_defaults (self):
        sdist.add_defaults(self)
        if self.distribution.has_f_libraries():
            build_flib = self.get_finalized_command('build_flib')
            self.filelist.extend(build_flib.get_source_files())

        if self.distribution.has_data_files():
            self.filelist.extend(self.distribution.get_data_files())
