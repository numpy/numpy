import os
import os.path
from os.path import join as pjoin, dirname as pdirname

#from distutils.core import build_py as old_build_py
from distutils.command.build_ext import build_ext as old_build_py
from numpy.distutils.ccompiler import CCompiler

# XXX: this is super ugly. The object/source filenames generations is handled
# inside compiler classes and build_ext in distutils, so to get the same
# convention, we derive scons command from build_ext instead of just Command.
class scons(old_build_py):
    description = "Scons builder"
    user_options = []

    def initialize_options(self):
        old_build_py.initialize_options(self)
        pass

    def finalize_options(self):
        old_build_py.finalize_options(self)
        if self.distribution.has_scons_scripts():
            print "Got it: scons scripts are %s" % self.distribution.scons_scripts
            self.scons_scripts = self.distribution.scons_scripts
        #        build_py = self.get_finalized_command('build_py')
        #print "!!!!!!!!!!!!!!!!!!"
        #print self.build_temp
        #print self.build_lib
        #print self.package

    def run(self):
        # XXX: when a scons script is missing, scons only prints warnings, and
        # does not return a failure (status is 0). We have to detect this from
        # distutils (this cannot work for recursive scons builds...)
        for i in self.scons_scripts:
            cmd = "scons -f " + i + ' -I. '
            cmd += ' src_dir=%s ' % pdirname(i)
            cmd += ' distutils_libdir=%s ' % pjoin(self.build_lib, pdirname(i))
            #print cmd
            st = os.system(cmd)
            if st:
                print "status is %d" % st
