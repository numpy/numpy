from distutils.core import Command

class scons(Command):
    description = "Scons builder"
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        if self.distribution.has_scons_scripts():
            print "Got it: scons scripts are %s" % self.distribution.scons_scripts
            self.scons_scripts = self.distribution.scons_scripts

    def run(self):
        # XXX: when a scons script is missing, scons only prints warnings, and
        # does not return a failure (status is 0). We have to detect this from
        # distutils (this cannot work for recursive scons builds...)
        for i in self.scons_scripts:
            cmd = "scons -f " + i + ' -I. '
            import os
            st = os.system(cmd)
            print "status is %d" % st
