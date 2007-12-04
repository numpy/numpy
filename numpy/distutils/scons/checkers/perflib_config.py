#! /usr/bin/env python
# Last Change: Tue Dec 04 03:00 PM 2007 J
from os.path import join as pjoin

from numpy.distutils.system_info import default_lib_dirs

from support import save_and_set, restore, ConfigOpts, ConfigRes

#------------------------
# Generic functionalities
#------------------------
class PerflibConfig:
    def __init__(self, name, section, defopts, headers, funcs, version_checker = None):
        """Initialize the configuration.

        Args:
            - name : str
                the name of the perflib
            - section : str
                the name of the section used in site.cfg for customization
            - defopts : ConfigOpts
                the compilation configuration for the checker
            - headers : list
                the list of headers to test in the checker
            - funcs : list
                the list of functions to test in the checker.
            - version_checker : callable
                optional function to check version of the perflib. Its
                arguments should be env and opts, where env is a scons
                environment and opts a ConfigOpts instance. It should return an
                integer (1 if successfull) and a version string."""
                
        self.name = name
        self.section = section
        self.defopts = defopts
        self.headers = headers
        self.funcs = funcs
        self.version_checker = version_checker

#-------------------------------------------
# Perflib specific configuration and helpers
#-------------------------------------------
CONFIG = {
        'MKL': PerflibConfig('MKL', 'mkl', ConfigOpts(libs = ['mkl', 'guide', 'm']),
                             ['mkl.h'], ['MKLGetVersion']),
        'ATLAS': PerflibConfig('ATLAS', 'atlas', 
                               ConfigOpts(libs = ['atlas'], 
                               libpath = [pjoin(i, 'atlas') for i in 
                                          default_lib_dirs]),
                               ['atlas_enum.h'],
                               ['ATL_sgemm']),
        'Accelerate' : PerflibConfig('Framework: Accelerate', 'accelerate', 
                                      ConfigOpts(frameworks = ['Accelerate']),
                                      ['Accelerate/Accelerate.h'],
                                      ['cblas_sgemm']),
        'vecLib' : PerflibConfig('Framework: vecLib', 'vecLib', 
                                 ConfigOpts(frameworks = ['vecLib']),
                                 ['vecLib/vecLib.h'],
                                 ['cblas_sgemm']),
        'Sunperf' : PerflibConfig('Sunperf', 'sunperf', 
                                  ConfigOpts(cflags = ['-dalign'], 
                                             linkflags = ['-xlic_lib=sunperf']),
                                  ['sunperf.h'],
                                  ['cblas_sgemm']),
        'FFTW3' : PerflibConfig('FFTW3', 'fftw', ConfigOpts(libs = ['fftw3']),
                                ['fftw3.h'], ['fftw_cleanup']),
        'FFTW2' : PerflibConfig('FFTW2', 'fftw', ConfigOpts(libs = ['fftw']),
                                ['fftw.h'], ['fftw_forget_wisdom'])
        }

class IsFactory:
    def __init__(self, name):
        """Name should be one key of CONFIG."""
        try:
            CONFIG[name]
        except KeyError, e:
            raise RuntimeError("name %s is unknown")

        def f(env, libname):
            if env['NUMPY_PKG_CONFIG'][libname] is None:
                return 0 == 1
            else:
                return env['NUMPY_PKG_CONFIG'][libname].name == \
                       CONFIG[name].name
        self.func = f

    def get_func(self):
        return self.func

class GetVersionFactory:
    def __init__(self, name):
        """Name should be one key of CONFIG."""
        try:
            CONFIG[name]
        except KeyError, e:
            raise RuntimeError("name %s is unknown")

        def f(env, libname):
            if env['NUMPY_PKG_CONFIG'][libname] is None or \
               not env['NUMPY_PKG_CONFIG'][libname].name == CONFIG[name].name:
                return 'No version info'
            else:
                return env['NUMPY_PKG_CONFIG'][libname].version
        self.func = f

    def get_func(self):
        return self.func

