import sys
if 'setuptools' in sys.modules:
    import setuptools.command.install as old_install_mod
else:
    import distutils.command.install as old_install_mod
old_install = old_install_mod.install
from distutils.file_util import write_file

class install(old_install):

    def finalize_options (self):
        old_install.finalize_options(self)
        self.install_lib = self.install_libbase

    def run(self):
        r = old_install.run(self)
        if self.record:
            # bdist_rpm fails when INSTALLED_FILES contains
            # paths with spaces. Such paths must be enclosed
            # with double-quotes.
            f = open(self.record,'r')
            lines = []
            need_rewrite = False
            for l in f.readlines():
                l = l.rstrip()
                if ' ' in l:
                    need_rewrite = True
                    l = '"%s"' % (l)
                lines.append(l)
            f.close()
            if need_rewrite:
                self.execute(write_file,
                             (self.record, lines),
                             "re-writing list of installed files to '%s'" %
                             self.record)
        return r
