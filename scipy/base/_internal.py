
from multiarray import _flagdict

_defflags = _flagdict.keys()

_setable = ['WRITEABLE', 'NOTSWAPPED', 'UPDATEIFCOPY', 'ALIGNED']
_setable2 = ['write','swap','uic','align']

class flagsobj(dict):
    def __init__(self, arr, flags):
        self._arr = arr
        self._flagnum = flags
        for k in _defflags:
            num = _flagdict[k]
            dict.__setitem__(self, k, flags & num == num)
        
    def __setitem__(self, item, val):
        val = not not val  # convert to boolean
        if item not in _setable:
            raise KeyError, "Cannot set that flag."
        dict.__setitem__(self, item, val) # Does this matter?

        kwds = {}
        for k, name in enumerate(_setable):
            if item == name:
                kwds[_setable2[k]] = val
        if (item == 'NOTSWAPPED'):
            kwds['swap'] = not val

        # now actually update array flags
        self._arr.setflags(**kwds)
        
