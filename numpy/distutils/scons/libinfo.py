#! /usr/bin/env python
# Last Change: Thu Oct 25 01:00 PM 2007 J

# Module for support to look for external code (replacement of
# numpy.distutils.system_info). KEEP THIS INDEPENDANT OF SCONS !
import os
import ConfigParser

from numpy.distutils.system_info import default_lib_dirs, \
    default_include_dirs, default_src_dirs, get_standard_file

def get_config():
    """ This tries to read .cfg files in several locations, and merge its
    information into a ConfigParser object for the first found file.
    
    Returns the ConfigParser instance. This copies the logic in system_info
    from numpy.distutils."""
    # Below is the feature we are copying from numpy.distutils:
    # 
    # The file 'site.cfg' is looked for in

    # 1) Directory of main setup.py file being run.
    # 2) Home directory of user running the setup.py file as ~/.numpy-site.cfg
    # 3) System wide directory (location of this file...)

    # The first one found is used to get system configuration options The
    # format is that used by ConfigParser (i.e., Windows .INI style). The
    # section DEFAULT has options that are the default for each section. The
    # available sections are fftw, atlas, and x11. Appropiate defaults are
    # used if nothing is specified.

    section = 'DEFAULT'
    defaults = {}
    defaults['libraries'] = ''
    defaults['library_dirs'] = os.pathsep.join(default_lib_dirs)
    defaults['include_dirs'] = os.pathsep.join(default_include_dirs)
    defaults['src_dirs'] = os.pathsep.join(default_src_dirs)
    cp = ConfigParser.ConfigParser(defaults)
    files = []
    files.extend(get_standard_file('.numpy-site.cfg'))
    files.extend(get_standard_file('site.cfg'))

    def parse_config_files():
        cp.read(files)
        if not cp.has_section(section):
            cp.add_section(section)

    parse_config_files()
    return cp, files

def parse_config_param(var):
    """Given var, the output of ConfirParser.get(section, name), returns a list
    of each item of its content."""
    varl = var.split(',')
    return [i.strip() for i in varl]

def get_paths(var):
    """Given var, the output of ConfirParser.get(section, name), returns a list
    of each item of its content, assuming the content is a list of directoris.
    
    Example: if var is foo:bar, it will return ['foo', 'bar'] on posix."""
    return var.split(os.pathsep)

if __name__ == '__main__':
    pass
