"""
    I don't know much about this one, so I'm not going to mess with 
    it much. (eric)
"""
from distutils.command.install import *
from distutils.command.install_headers import install_headers as old_install_headers

class install_headers (old_install_headers):
    def run (self):
        headers = self.distribution.headers
        if not headers:
            return
        # hack to force headers into Numeric instead of SciPy
        import os
        d,f = os.path.split(self.install_dir)
        self.install_dir = os.path.join(d,'Numeric')        
        self.mkpath(self.install_dir)
        for header in headers:
            (out, _) = self.copy_file(header, self.install_dir)
            self.outfiles.append(out)    
