""" Override the develop command from setuptools so we can ensure that build_src
--inplace gets executed.
"""

from setuptools.command.develop import develop as old_develop

class develop(old_develop):
    __doc__ = old_develop.__doc__
    def install_for_development(self):
        # Build sources in-place, too.
        self.reinitialize_command('build_src', inplace=1)
        old_develop.install_for_development(self)
