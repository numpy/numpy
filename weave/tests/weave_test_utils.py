import os,sys,string
import pprint 

def remove_whitespace(in_str):
    import string
    out = string.replace(in_str," ","")
    out = string.replace(out,"\t","")
    out = string.replace(out,"\n","")
    return out
    
def print_assert_equal(test_string,actual,desired):
    """this should probably be in scipy_test
    """
    try:
        assert(actual == desired)
    except AssertionError:
        import cStringIO
        msg = cStringIO.StringIO()
        msg.write(test_string)
        msg.write(' failed\nACTUAL: \n')
        pprint.pprint(actual,msg)
        msg.write('DESIRED: \n')
        pprint.pprint(desired,msg)
        raise AssertionError, msg.getvalue()

###################################################
# mainly used by catalog tests               
###################################################
from scipy_distutils.misc_util import add_grandparent_to_path,restore_path

add_grandparent_to_path(__name__)
import catalog
restore_path()

def temp_catalog_files():
    # might need to add some more platform specific catalog file
    # suffixes to remove.  The .pag was recently added for SunOS
    d = catalog.default_dir()
    f = catalog.os_dependent_catalog_name()
    suffixes = ['.dat','.dir','.pag','']
    cat_files = [os.path.join(d,f+suffix) for suffix in suffixes]
    return cat_files

def clear_temp_catalog():
    """ Remove any catalog from the temp dir
    """
    cat_files = temp_catalog_files()
    for catalog_file in cat_files:
        if os.path.exists(catalog_file):
            if os.path.exists(catalog_file+'.bak'):
                os.remove(catalog_file+'.bak')
            os.rename(catalog_file,catalog_file+'.bak')

def restore_temp_catalog():
    """ Remove any catalog from the temp dir
    """
    cat_files = temp_catalog_files()
    for catalog_file in cat_files:
        if os.path.exists(catalog_file+'.bak'):
            if os.path.exists(catalog_file):            
                os.remove(catalog_file)
            os.rename(catalog_file+'.bak',catalog_file)

def empty_temp_dir():
    """ Create a sub directory in the temp directory for use in tests
    """
    import tempfile
    d = catalog.default_dir()
    for i in range(10000):
        new_d = os.path.join(d,tempfile.gettempprefix()[1:-1]+`i`)
        if not os.path.exists(new_d):
            os.mkdir(new_d)
            break
    return new_d

def cleanup_temp_dir(d):
    """ Remove a directory created by empty_temp_dir
        should probably catch errors
    """
    files = map(lambda x,d=d: os.path.join(d,x),os.listdir(d))
    for i in files:
        try:
            if os.path.isdir(i):
                cleanup_temp_dir(i)
            else:
                os.remove(i)
        except OSError:
            pass # failed to remove file for whatever reason 
                 # (maybe it is a DLL Python is currently using)        
    try:
        os.rmdir(d)
    except OSError:
        pass        