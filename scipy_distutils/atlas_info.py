import sys, os
from misc_util import get_path

library_path = ''

def get_atlas_info():
    if sys.platform  == 'win32':
        if not library_path:
            atlas_library_dirs=['C:\\atlas\\WinNT_PIIISSE1']
        else:
            atlas_library_dirs = library_path
        blas_libraries = ['f77blas', 'cblas', 'atlas', 'g2c']
        lapack_libraries = ['lapack'] + blas_libraries 
    else:
        if not library_path:
            atlas_library_dirs = unix_atlas_directory(sys.platform)
        else:
            atlas_library_dirs = library_path
        blas_libraries = ['cblas','f77blas','atlas']
        lapack_libraries = ['lapack'] + blas_libraries
    return blas_libraries, lapack_libraries, atlas_library_dirs

def unix_atlas_directory(platform):
    """ Search a list of common locations looking for the atlas directory.
 
        Return None if the directory isn't found, otherwise return the
        directory name.  This isn't very sophisticated right now.  I can
        imagine doing an ftp to our server on platforms that we know about.
 
        Atlas is a highly optimized version of lapack and blas that is fast
        on almost all platforms.
    """
    result = [] #None
    # do a little looking for the linalg directory for atlas libraries
    #path = get_path(__name__)
    #local_atlas0 = os.path.join(path,platform,'atlas')
    #local_atlas1 = os.path.join(path,platform[:-1],'atlas')
 
    # first look for a system defined atlas directory
    dir_search = ['/usr/local/lib/atlas','/usr/lib/atlas']#,
    #              local_atlas0, local_atlas1]
    for directory in dir_search:
        if os.path.exists(directory):
            result = [directory]

    # we should really do an ftp search or something like that at this point.
    return result   
