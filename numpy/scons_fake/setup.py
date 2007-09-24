import os
import os.path

def get_object_names(source_filenames, strip_dir=0, output_dir=''):
    # ripped off distutilc.ccompiler (CCompiler_object_filenames)
    if output_dir is None:
        output_dir = ''
    obj_names = []
    for src_name in source_filenames:
        base, ext = os.path.splitext(os.path.normpath(src_name))
        base = os.path.splitdrive(base)[1] # Chop off the drive
        base = base[os.path.isabs(base):]  # If abs, chop off leading /
        if base.startswith('..'):
            # Resolve starting relative path components, middle ones
            # (if any) have been handled by os.path.normpath above.
            i = base.rfind('..')+2
            d = base[:i]
            d = os.path.basename(os.path.abspath(d))
            base = d + base[i:]
        #XXX: how to know which file types are supported ?
        #if ext not in self.src_extensions:
        #    raise UnknownFileError, \
        #          "unknown file type '%s' (from '%s')" % (ext, src_name)
        if strip_dir:
            base = os.path.basename(base)
        #XXX: change '.o' to something like obj_extension 
        obj_name = os.path.join(output_dir,base + '.o')
        obj_names.append(obj_name)
    return obj_names

def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    from numpy.distutils.system_info import get_info
    config = Configuration('scons_fake',parent_package,top_path)

    #config.add_library('_fortran_foo',
    #                   sources=['foo.f'])
    config.add_sconscript('SConstruct')
    config.add_data_dir('tests')
    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(configuration=configuration)
