from distutils.core import *
from distutils.core import setup as old_setup

from distutils.cmd import Command
from scipy_distutils.extension import Extension

# Our dist is different than the standard one.
from scipy_distutils.dist import Distribution

from scipy_distutils.command import build
from scipy_distutils.command import build_py
from scipy_distutils.command import build_ext
from scipy_distutils.command import build_clib
from scipy_distutils.command import build_flib
from scipy_distutils.command import run_f2py
from scipy_distutils.command import sdist
from scipy_distutils.command import install_data
from scipy_distutils.command import install
from scipy_distutils.command import install_headers

def setup(**attr):
    distclass = Distribution
    cmdclass = {'build':            build.build,
                'build_flib':       build_flib.build_flib,
                'build_ext':        build_ext.build_ext,
                'build_py':         build_py.build_py,                
                'build_clib':       build_clib.build_clib,
                'run_f2py':       run_f2py.run_f2py,
                'sdist':            sdist.sdist,
                'install_data':     install_data.install_data,
                'install':          install.install,
                'install_headers':  install_headers.install_headers
                }
                      
    new_attr = attr.copy()
    if new_attr.has_key('cmdclass'):
        cmdclass.update(new_attr['cmdclass'])        
    new_attr['cmdclass'] = cmdclass
    
    if not new_attr.has_key('distclass'):
        new_attr['distclass'] = distclass    
    
    return old_setup(**new_attr)
