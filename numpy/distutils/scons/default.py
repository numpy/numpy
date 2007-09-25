def tool_list(platform):
    """platform should be the value returned by enbv['PLATFORM'], not
    sys.platform !!!!."""
    # Here, we set the list of default tools as used by numpy.distutils.scons. This
    # is ripped of scons, because scons does not provide a way to get separate
    # default lists for separate tools (e.g linker, C compiler, etc...)

    if str(platform) == 'win32':
        "prefer Microsoft tools on Windows"
        linkers = ['mslink', 'gnulink', 'ilink', 'linkloc', 'ilink32' ]
        c_compilers = ['msvc', 'mingw', 'gcc', 'intelc', 'icl', 'icc', 'cc', 'bcc32' ]
        cxx_compilers = ['msvc', 'intelc', 'icc', 'g++', 'c++', 'bcc32' ]
        assemblers = ['masm', 'nasm', 'gas', '386asm' ]
        fortran_compilers = ['g77', 'ifl', 'cvf', 'f95', 'f90', 'fortran']
        ars = ['mslib', 'ar', 'tlib']
    elif str(platform) == 'os2':
        "prefer IBM tools on OS/2"
        linkers = ['ilink', 'gnulink', 'mslink']
        c_compilers = ['icc', 'gcc', 'msvc', 'cc']
        cxx_compilers = ['icc', 'g++', 'msvc', 'c++']
        assemblers = ['nasm', 'masm', 'gas']
        fortran_compilers = ['ifl', 'g77']
        ars = ['ar', 'mslib']
    elif str(platform) == 'irix':
        "prefer MIPSPro on IRIX"
        linkers = ['sgilink', 'gnulink']
        c_compilers = ['sgicc', 'gcc', 'cc']
        cxx_compilers = ['sgic++', 'g++', 'c++']
        assemblers = ['as', 'gas']
        fortran_compilers = ['f95', 'f90', 'f77', 'g77', 'fortran']
        ars = ['sgiar']
    elif str(platform) == 'sunos':
        "prefer Forte tools on SunOS"
        linkers = ['sunlink', 'gnulink']
        c_compilers = ['suncc', 'gcc', 'cc']
        cxx_compilers = ['sunc++', 'g++', 'c++']
        assemblers = ['as', 'gas']
        fortran_compilers = ['f95', 'f90', 'f77', 'g77', 'fortran']
        ars = ['sunar']
    elif str(platform) == 'hpux':
        "prefer aCC tools on HP-UX"
        linkers = ['hplink', 'gnulink']
        c_compilers = ['hpcc', 'gcc', 'cc']
        cxx_compilers = ['hpc++', 'g++', 'c++']
        assemblers = ['as', 'gas']
        fortran_compilers = ['f95', 'f90', 'f77', 'g77', 'fortran']
        ars = ['ar']
    elif str(platform) == 'aix':
        "prefer AIX Visual Age tools on AIX"
        linkers = ['aixlink', 'gnulink']
        c_compilers = ['aixcc', 'gcc', 'cc']
        cxx_compilers = ['aixc++', 'g++', 'c++']
        assemblers = ['as', 'gas']
        fortran_compilers = ['f95', 'f90', 'aixf77', 'g77', 'fortran']
        ars = ['ar']
    elif str(platform) == 'darwin':
        "prefer GNU tools on Mac OS X, except for some linkers and IBM tools"
        linkers = ['applelink', 'gnulink']
        c_compilers = ['gcc', 'cc']
        cxx_compilers = ['g++', 'c++']
        assemblers = ['as']
        fortran_compilers = ['f95', 'f90', 'g77']
        ars = ['ar']
    else:
        "prefer GNU tools on all other platforms"
        linkers = ['gnulink', 'mslink', 'ilink']
        c_compilers = ['gcc', 'msvc', 'intelc', 'icc', 'cc']
        cxx_compilers = ['g++', 'msvc', 'intelc', 'icc', 'c++']
        assemblers = ['gas', 'nasm', 'masm']
        fortran_compilers = ['f95', 'f90', 'g77', 'ifort', 'ifl', 'fortran']
        ars = ['ar', 'mslib']

    other_tools = ['BitKeeper', 'CVS', 'dmd', 'dvipdf', 'dvips', 'gs', 'jar',
            'javac', 'javah', 'latex', 'lex', 'm4', 'midl', 'msvs', 'pdflatex',
            'pdftex', 'Perforce', 'RCS', 'rmic', 'rpcgen', 'SCCS', 'swig',
            'tar', 'tex', 'yacc', 'zip']
    return linkers, c_compilers, cxx_compilers, assemblers, fortran_compilers, \
           ars, other_tools
