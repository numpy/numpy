from os.path import join as pjoin

def NumpySharedLibrary(env, target, source, *args, **kw):
    """This builder is the same than SharedLibrary, except for the fact that it
    takes into account build dir info passed by distutils, and put the target at
    the right location in distutils build directory for correct installation."""
    source = [pjoin(env['build_dir'], i) for i in source]
    # XXX: why target is a list ? It is always true ?
    lib = env.SharedLibrary("$build_dir/%s" % target[0], source, *args, **kw)

    inst_lib = env.Install("$distutils_installdir", lib)
    return lib, inst_lib
	
def NumpyPythonExtension(env, target, source, *args, **kw):
    """This builder is the same than PythonExtension, except for the fact that it
    takes into account build dir info passed by distutils, and put the target at
    the right location in distutils build directory for correct installation."""
    import SCons.Util
    newsource = []
    for i in source:
        if SCons.Util.is_String(i):
            newsource.append(pjoin(env['build_dir'], i)) 
        else:
            newsource.append(i) 
    # XXX: why target is a list ? It is always true ?
    lib = env.PythonExtension("$build_dir/%s" % target[0], newsource, *args, **kw)

    inst_lib = env.Install("$distutils_installdir", lib)
    return lib, inst_lib
	
	
def NumpyCtypes(env, target, source, *args, **kw):
    """This builder is essentially the same than SharedLibrary, but should be
    used for libraries which will only be used through ctypes.

    In particular, it does not install .exp/.lib files on windows. """
    source = [pjoin(env['build_dir'], i) for i in source]

    # XXX: why target is a list ? It is always true ?
    # XXX: handle cases where SHLIBPREFIX is in args
    lib = env.SharedLibrary("$build_dir/%s" % target[0], 
                            source, 
                            SHLIBPREFIX = '', 
                            *args, 
                            **kw)
    lib = [i for i in lib if not (str(i).endswith('.exp') or str(i).endswith('.lib')) ]
    inst_lib = env.Install("$distutils_installdir", lib)
    return lib, inst_lib

def NumpyFromCTemplate(env, target, source, *args, **kw):
    source = [pjoin(env['build_dir'], i) for i in source]

    # XXX: why target is a list ? It is always true ?
    # XXX: handle cases where SHLIBPREFIX is in args
    src = env.FromCTemplate("$build_dir/%s" % target[0], 
                            source, *args, **kw)

    #inst_src = env.Install("$distutils_installdir", src)
    #return src, inst_src
    return src

def NumpyFromFTemplate(env, target, source, *args, **kw):
    source = [pjoin(env['build_dir'], i) for i in source]

    # XXX: why target is a list ? It is always true ?
    # XXX: handle cases where SHLIBPREFIX is in args
    src = env.FromFTemplate("$build_dir/%s" % target[0], 
                            source, *args, **kw)

    #inst_src = env.Install("$distutils_installdir", src)
    #return src, inst_src
    return src
