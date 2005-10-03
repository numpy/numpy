
from multiarray import _flagdict

_defflags = _flagdict.keys()

_setable = ['WRITEABLE', 'NOTSWAPPED', 'UPDATEIFCOPY', 'ALIGNED',
            'W','N','U','A']
_setable2 = ['write','swap','uic','align']*2
_firstltr = {'W':'WRITEABLE',
             'N':'NOTSWAPPED',
             'A':'ALIGNED',
             'C':'CONTIGUOUS',
             'F':'FORTRAN',
             'O':'OWNDATA',
             'U':'UPDATEIFCOPY'}

_anum = _flagdict['ALIGNED']
_nnum = _flagdict['NOTSWAPPED']
_wnum = _flagdict['WRITEABLE']
_cnum = _flagdict['CONTIGUOUS']
_fnum = _flagdict['FORTRAN']

class flagsobj(dict):
    def __init__(self, arr, flags):
        self._arr = arr
        self._flagnum = flags
        for k in _defflags:
            num = _flagdict[k]
            dict.__setitem__(self, k, flags & num == num)

    def __getitem__(self, key):
        if not isinstance(key, str):
            raise KeyError, "Unknown flag %s" % key
        if len(key) == 1:
            try:
                return dict.__getitem__(self, _firstltr[key])
            except:
                if (key == 'B'):
                    num = _anum + _nnum + _wnum
                    return self._flagnum & num == num
                elif (key == 'S'):
                    return not (self._flagnum & _nnum == _nnum)
        else:
            try:
                return dict.__getitem__(self, key)
            except: # special cases
                if (key == 'FNC'):
                    return (self._flagnum & _fnum == _fnum) and not \
                           (self._flagnum & _cnum == _cnum)
	        if (key == 'FORC'):
		    return (self._flagnum & _fnum == _fnum) or \
                           (self._flagnum & _cnum == _cnum)
                if (key == 'SWAPPED'):
                    return not (self._flagnum & _nnum == _nnum)
                if (key == 'BEHAVED'):
                    num = _anum + _nnum + _wnum
                    return self._flagnum & num == num
                if (key in ['BEHAVED_RO', 'BRO']):
                    num = _anum + _nnum
                    return self._flagnum & num == num
                if (key in ['CARRAY','CA']):
                    num = _anum + _nnum + _wnum + _cnum
                    return self._flagnum & num == num
                if (key in ['FARRAY','FA']):
                    num = _anum + _nnum + _wnum + _fnum
                    return (self._flagnum & num == num) and not \
                           (self._flagnum & _cnum == _cnum)
        raise KeyError, "Unknown flag: %s" % key
        
    def __setitem__(self, item, val):
        val = not not val  # convert to boolean
        if item not in _setable:
            raise KeyError, "Cannot set flag", item
        dict.__setitem__(self, item, val) # Does this matter?

        kwds = {}
        for k, name in enumerate(_setable):
            if item == name:
                kwds[_setable2[k]] = val
        if (item == 'NOTSWAPPED'):
            kwds['swap'] = not val

        # now actually update array flags
        self._arr.setflags(**kwds)
        
