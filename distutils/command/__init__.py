"""distutils.command

Package containing implementation of all the standard Distutils
commands."""

__revision__ = "$Id$"

distutils_all = [  'build_py',
                   'build_scripts',
                   'clean',
                   'install_lib',
                   'install_scripts',
                   'bdist',
                   'bdist_dumb',
                   'bdist_wininst',
                ]

__import__('distutils.command',globals(),locals(),distutils_all)

__all__ = ['build',
           'config_compiler',
           'build_src',
           'build_ext',
           'build_clib',
           'install',
           'install_data',
           'install_headers',
           'bdist_rpm',
           'sdist',
          ] + distutils_all
