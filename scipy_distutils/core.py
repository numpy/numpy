from distutils.core import *
from distutils.core import setup as old_setup

from distutils.cmd import Command
from distutils.extension import Extension

# Our dist is different than the standard one.
from dist import Distribution

import command.build
import command.build_ext
import command.build_clib
import command.build_flib
import command.sdist
import command.install_data
import command.install
import command.install_headers

def setup(**attr):
    distclass = Distribution
    cmdclass = {'build':command.build.build,
                'build_flib':command.build_flib.build_flib,
                'build_ext':command.build_ext.build_ext,
                'build_clib':command.build_clib.build_clib,
                'sdist':command.sdist.sdist,
                'install_data': command.install_data.install_data,
                'install':command.install.install,
                'install_headers': command.install_headers.install_headers}
                      
    new_attr = attr.copy()
    if new_attr.has_key('cmdclass'):
        cmdclass.update(new_attr['cmdclass'])        
    new_attr['cmdclass'] = cmdclass
    
    if not new_attr.has_key('distclass'):
        new_attr['distclass'] = distclass    
    
    return old_setup(**new_attr)
