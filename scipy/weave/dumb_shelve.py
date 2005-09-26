from shelve import Shelf
import zlib
from cStringIO import  StringIO
import  cPickle  
import dumbdbm_patched

class DbfilenameShelf(Shelf):
    """Shelf implementation using the "anydbm" generic dbm interface.

    This is initialized with the filename for the dbm database.
    See the module's __doc__ string for an overview of the interface.
    """
    
    def __init__(self, filename, flag='c'):
        Shelf.__init__(self, dumbdbm_patched.open(filename, flag))

    def __getitem__(self, key):
        compressed = self.dict[key]
        try:
            r = zlib.decompress(compressed)
        except zlib.error:
            r = compressed
        return cPickle.loads(r) 
        
    def __setitem__(self, key, value):
        s = cPickle.dumps(value,1)
        self.dict[key] = zlib.compress(s)

def open(filename, flag='c'):
    """Open a persistent dictionary for reading and writing.

    Argument is the filename for the dbm database.
    See the module's __doc__ string for an overview of the interface.
    """
    
    return DbfilenameShelf(filename, flag)
