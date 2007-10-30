# XXX: Handle setuptools ?
from distutils.core import Distribution

# This class is used because we add new files (sconscripts, and so on) with the
# scons command
class NumpyDistribution(Distribution):
    def __init__(self, attrs = None):
        # A list of (sconscripts, pre_hook, post_hook, src)
        self.scons_data = []
        Distribution.__init__(self, attrs)

    def has_scons_scripts(self):
        return bool(self.scons_data)

    def get_scons_scripts(self):
        return [i[0] for i in self.scons_data]

    def get_scons_pre_hooks(self):
        return [i[1] for i in self.scons_data]

    def get_scons_post_hooks(self):
        return [i[2] for i in self.scons_data]

    def get_scons_sources(self):
        return [i[3] for i in self.scons_data]

