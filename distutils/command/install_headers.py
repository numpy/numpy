import os
from distutils.command.install import *
from distutils.command.install_headers import install_headers as old_install_headers

class install_headers (old_install_headers):

    def run (self):
        headers = self.distribution.headers
        if not headers:
            return

        prefix = os.path.dirname(self.install_dir)        
        for header in headers:
            if isinstance(header,tuple):
                d = os.path.join(*([prefix]+header[0].split('.')))
                header = header[1]
            else:
                d = self.install_dir
            self.mkpath(d)
            (out, _) = self.copy_file(header, d)
            self.outfiles.append(out)
