
import os
import imp
from glob import glob

def import_packages(rootpath):
    """ Import packages in the current directory that implement
    info.py. See DEVELOPERS.txt for more info.
    """
    for info_file in glob(os.path.join(rootpath,'*','info.py')):
         package_name = os.path.basename(os.path.dirname(info_file))
         print info_file,package_name
         continue
         try:
             info_module = imp.load_module()
         except Exception,msg:
             print msg
             info_module = None
