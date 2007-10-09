#! /usr/bin/env python
# Last Change: Tue Oct 09 04:00 PM 2007 J

# Module for support to look for external code (replacement of
# numpy.distutils.system_info). scons dependant code.
import ConfigParser
from copy import deepcopy

from SCons.Util import is_List

from libinfo import get_config, get_paths
from utils import get_empty

_SYMBOL_DEF_STR = """
#ifdef __cplusplus
extern "C"
#endif
char %s();\n"""

_MAIN_CALL_CENTER = """
int main(int argc, char** argv)
{
    %s
    return 0;
}\n """

def _CheckLib(context, libs, symbols, header, language, section, siteconfig, 
              libpath, cpppath):
    """Implementation for checking a list of symbols, with libs.
    
    Assumes that libs, symbol, header, libpath and cpppath are sequences (list
    or tuples). DO NOT USE DIRECTLY IN SCONSCRIPT !!!"""
    # XXX: sanitize API for section/siteconfig option: if sectionis not given,
    # can we just say to ignore the sitecfg ?
    if not siteconfig:
        siteconfig, cfgfiles = get_config()

    def get_descr():
        descr = siteconfig.items(section)
        sdescr = ""
        for i in descr:
            sdescr += str(i) + '\n'
        return sdescr
        
    # Generate the source string of the conftest
    src = ""
    callstr = ""

    if symbols:
        for s in symbols:
            # XXX: should put undef here (ala autoconf)
            src += _SYMBOL_DEF_STR % s
            callstr += "%s();" % s

    src += _MAIN_CALL_CENTER % callstr
    # HUGE HACK: we want this test to depend on site.cfg files obviously, since
    # a change in them can change the libraries tested. But Depends does not
    # seem to work in configuration context, and I don't see any simple way to
    # have the same functionality. So I put the configuration we got from
    # get_config into the source code, such as a change in site.cfg will change
    # the source file, and thus taken into account to decide whether to rebuild
    # from tjhe SconfTaskMaster point of view.

    # XXX: I put the content between #if 0 / #endif, which is the most portable
    # way I am aware of for multilines comments in C and C++ (this is also
    # recommended in C++ portability guide of mozilla for nested comments,
    # which may happen here). This is also the most robust, since it seems
    # unlikely to have any #endif somewhere in the return value of get_descr.
    #src += "#if 0\n"
    #src += get_descr()
    #src += "\n#endif\n"

    # XXX: handle autoadd
    # XXX: handle extension 
    extension = '.c'

    if section and siteconfig:
        #print "Checking %s from section %s" % (libs[0], section)
        res = _check_lib_section(context, siteconfig, section, src, libs,
                                 libpath, cpppath)
    else:
        oldLIBS = context.env.has_key('LIBS') and deepcopy(context.env['LIBS'])
        context.env.Append(LIBS = libs)
        res = context.TryLink(src, '.c')
        if not res:
            context.env.Replace(LIBS = oldLIBS) 

    return res


def NumpyCheckLib(context, libs, symbols = None, header = None, 
                  language = None, section = None, siteconfig = None, name = None):
    """Check for symbol in libs. 
    
    This is the general purpose replacement for numpy.distutils.system_info. It
    uses the options in siteconfig so that search path can be overwritten in
    *.cfg files (using section given by section argument). If siteconfig is
    None, it does uses get_config function to get the configuration, which
    gives the old numpy.distutils behaviour to get options.

    
    Convention: if the section has *dirs parameters, it will use them all in
    one pass, e.g if library_dirs is ['/usr/local/lib', '/usr/local/mylib'], it
    will try to link the given libraries by appending both directories to the
    LIBPATH."""
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # This is really preliminary, and needs a lot of love before being in good
    # shape !!!!!
    #
    # Biggest problem: how to show information about found libraries ? Since
    # they are found implicitely through build tools (compiler and linker), we
    # can not give explicit information. IMHO (David Cournapeau), it is better
    # to find them implicitely because it is much more robust. But for the info...
    # 
    # This needs testing, too.
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    env = context.env

    # XXX: would be nice for each extension to add an option to command line.
    # XXX: handle env var
    # XXX: handle language
    if language:
        raise NotImplementedError("FIXME: language selection not implemented yet !")

    # Make sure libs and symbols are lists
    if libs and not is_List(libs):
        libs = [libs]
    if symbols and not is_List(symbols):
        symbols = [symbols]

    if not name:
        name = libs[0]

    # Display message
    if symbols:
        sbstr = ', '.join(symbols)
        context.Message('Checking for symbol(s) %s in %s... ' % (sbstr, name))
    else:
        context.Message('Checking for %s... ' % name)

    # Call the implementation
    libpath = None
    cpppath = None
    res = _CheckLib(context, libs, symbols, header, language, section,
                    siteconfig, libpath, cpppath, )
    context.Result(res)
    return res

def _check_lib_section(context, siteconfig, section, src, libs, libpath, cpppath):
    # Convention: if an option is found in site.cfg for the given section, it
    # takes precedence on the arguments libs, libpath, cpppath.
    res = 1
    try:
        newLIBPATH = get_paths(siteconfig.get(section, 'library_dirs'))
    except ConfigParser.NoSectionError, e:
        if libpath:
            newLIBPATH = libpath
        else:
            newLIBPATH = []

    try:
        newCPPPATH = get_paths(siteconfig.get(section, 'include_dirs'))
    except ConfigParser.NoSectionError, e:
        if cpppath:
            newCPPPATH = cpppath
        else:
            newCPPPATH = []

    try:
        newLIBS = siteconfig.get(section, 'libraries') 
    except ConfigParser.NoSectionError, e:
        if libs:
            newLIBS = libs
        else:
            newLIBS = []

    lastLIBPATH = get_empty(context.env,'LIBPATH')
    lastLIBS = get_empty(context.env,'LIBS')
    lastCPPPATH = get_empty(context.env,'CPPPATH')
    context.env.Append(LIBPATH = newLIBPATH)
    context.env.Append(LIBS = newLIBS)
    context.env.Append(CPPPATH = newCPPPATH)
    res *= context.TryLink(src, '.c')
    if not res:
        context.env.Replace(LIBS = lastLIBS, 
                            LIBPATH = lastLIBPATH, 
                            CPPPATH = lastCPPPATH)

    return res
