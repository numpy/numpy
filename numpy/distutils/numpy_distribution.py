from distutils.core import Distribution

class NumpyDistribution(Distribution):
    def __init__(self, attrs = None):
        self.scons_scripts = []
        Distribution.__init__(self, attrs)

    def has_scons_scripts(self):
        return bool(self.scons_scripts)

    def get_scons_scripts(self):
        return [i[0] for i in self.scons_scripts]
