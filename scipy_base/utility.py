import Numeric
import types
import cPickle
import sys

from Numeric import *
from fastumath import *

ScalarType = [types.IntType, types.LongType, types.FloatType, types.ComplexType]

__all__ = ['grid','r_','c_','index_exp','disp','r1array','r2array','typename',
           'who','objsave','objload','isscalar','all_mat','select','atleast_1d',
           'atleast_2d','atleast_3d','vstack','hstack','column_stack','dstack',
           'replace_zero_by_x_arrays','array_split','split','hsplit','vsplit',
           'dsplit','array_kind','array_precision','array_type','common_type',
           'trim_zeros','cast']



toChar = lambda x: Numeric.array(x, Numeric.Character)
toInt8 = lambda x: Numeric.array(x, Numeric.Int8)# or use variable names such as Byte
toInt16 = lambda x: Numeric.array(x, Numeric.Int16)
toInt32 = lambda x: Numeric.array(x, Numeric.Int32)
toInt = lambda x: Numeric.array(x, Numeric.Int)
toFloat32 = lambda x: Numeric.array(x, Numeric.Float32)
toFloat64 = lambda x: Numeric.array(x, Numeric.Float64)
toComplex32 = lambda x: Numeric.array(x, Numeric.Complex32)
toComplex64 = lambda x: Numeric.array(x, Numeric.Complex64)

cast = {Numeric.Character: toChar,
        Numeric.Int8: toInt8,
        Numeric.Int16: toInt16,
        Numeric.Int32: toInt32,
        Numeric.Int: toInt,
        Numeric.Float32: toFloat32,
        Numeric.Float64: toFloat64,
        Numeric.Complex32: toComplex32,
        Numeric.Complex64: toComplex64}

class nd_grid:
    """Construct a "meshgrid" in N-dimensions.

    grid = nd_grid() creates an instance which will return a mesh-grid
    when indexed.  The dimension and number of the output arrays are equal
    to the number of indexing dimensions.  If the step length is not a complex
    number, then the stop is not inclusive.

    However, if the step length is a COMPLEX NUMBER (e.g. 5j), then the integer
    part of it's magnitude is interpreted as specifying the number of points to
    create between the start and stop values, where the stop value
    IS INCLUSIVE.

    Example:

       >>> grid = nd_grid()
       >>> grid[0:5,0:5]
       array([[[0, 0, 0, 0, 0],
               [1, 1, 1, 1, 1],
               [2, 2, 2, 2, 2],
               [3, 3, 3, 3, 3],
               [4, 4, 4, 4, 4]],
              [[0, 1, 2, 3, 4],
               [0, 1, 2, 3, 4],
               [0, 1, 2, 3, 4],
               [0, 1, 2, 3, 4],
               [0, 1, 2, 3, 4]]])
       >>> grid[-1:1:5j]
       array([-1. , -0.5,  0. ,  0.5,  1. ])
    """
    def __getitem__(self,key):
        try:
	    size = []
            typecode = Numeric.Int
	    for k in range(len(key)):
	        step = key[k].step
                start = key[k].start
                if start is None: start = 0
                if step is None:
                    step = 1
                if type(step) is type(1j):
                    size.append(int(abs(step)))
                    typecode = Numeric.Float
                else:
                    size.append(int((key[k].stop - start)/(step*1.0)))
                if isinstance(step,types.FloatType) or \
                   isinstance(start, types.FloatType) or \
                   isinstance(key[k].stop, types.FloatType):
                       typecode = Numeric.Float
            nn = Numeric.indices(size,typecode)
	    for k in range(len(size)):
                step = key[k].step
                if step is None:
                    step = 1
                if type(step) is type(1j):
                    step = int(abs(step))
                    step = (key[k].stop - key[k].start)/float(step-1)
                nn[k] = (nn[k]*step+key[k].start)
	    return nn
        except (IndexError, TypeError):
            step = key.step
            stop = key.stop
            start = key.start
            if start is None: start = 0
            if type(step) is type(1j):
                step = abs(step)
                length = int(step)
                step = (key.stop-start)/float(step-1)
                stop = key.stop+step
                return Numeric.arange(0,length,1,Numeric.Float)*step + start
            else:
                return Numeric.arange(start, stop, step)
	    
    def __getslice__(self,i,j):
        return Numeric.arange(i,j)

    def __len__(self):
        return 0

grid = nd_grid()

class concatenator:
    """An object which translates slice objects to concatenation along an axis.
    """
    def __init__(self, axis=0):
        self.axis = axis
    def __getitem__(self,key):
        if type(key) is not types.TupleType:
            key = (key,)
        objs = []
        for k in range(len(key)):
            if type(key[k]) is types.SliceType:
                typecode = Numeric.Int
	        step = key[k].step
                start = key[k].start
                stop = key[k].stop
                if start is None: start = 0
                if step is None:
                    step = 1
                if type(step) is type(1j):
                    size = int(abs(step))
                    typecode = Numeric.Float
                    endpoint = 1
                else:
                    size = int((stop - start)/(step*1.0))
                    endpoint = 0
                if isinstance(step,types.FloatType) or \
                   isinstance(start, types.FloatType) or \
                   isinstance(stop, types.FloatType):
                       typecode = Numeric.Float
                newobj = linspace(start, stop, num=size, endpoint=endpoint)
            elif type(key[k]) in ScalarType:
                newobj = Numeric.asarray([key[k]])
            else:
                newobj = key[k]
            objs.append(newobj)
        return Numeric.concatenate(tuple(objs),axis=self.axis)
        
    def __getslice__(self,i,j):
        return Numeric.arange(i,j)

    def __len__(self):
        return 0

r_=concatenator(0)
c_=concatenator(-1)


# A nicer way to build up index tuples for arrays.
#
# You can do all this with slice() plus a few special objects,
# but there's a lot to remember. This version is simpler because
# it uses the standard array indexing syntax.
#
# Written by Konrad Hinsen <hinsen@cnrs-orleans.fr>
# last revision: 1999-7-23
#
# Cosmetic changes by T. Oliphant 2001
#
#
# This module provides a convenient method for constructing
# array indices algorithmically. It provides one importable object,
# 'index_expression'.
#
# For any index combination, including slicing and axis insertion,
# 'a[indices]' is the same as 'a[index_expression[indices]]' for any
# array 'a'. However, 'index_expression[indices]' can be used anywhere
# in Python code and returns a tuple of slice objects that can be
# used in the construction of complex index expressions.

class _index_expression_class:

    maxint = sys.maxint

    def __getitem__(self, item):
        if type(item) != type(()):
            return (item,)
        else:
            return item

    def __len__(self):
        return self.maxint

    def __getslice__(self, start, stop):
        if stop == self.maxint:
            stop = None
        return self[start:stop:None]

index_exp = _index_expression_class()

# End contribution from Konrad.

def disp(mesg, device=None, linefeed=1):
    """Display a message to device (default is sys.stdout) with(out) linefeed.
    """
    if device is None:
        device = sys.stdout
    if linefeed:
        device.write('%s\n' % mesg)
    else:
        device.write('%s' % mesg)
    device.flush()
    return


def r1array(x):
    """Ensure x is at least 1-dimensional.
    """
    if type(x) in [type(1.0), type(1L), type(1), type(1j)]:
        x = [x]
    elif (type(x) is Numeric.ArrayType) and (len(x.shape) == 0):
        x.shape = (1,)
    return Numeric.asarray(x)

def r2array(x):
    """Ensure x is at least 2-dimensional.
    """
    if type(x) in [type(1.0), type(1L), type(1), type(1j)]:
        return Numeric.asarray([[x]])
    else:
        temp = Numeric.asarray(x)
    if len(temp.shape) == 1:
        if temp.shape[0] == 0:
            temp.shape = (0,) + temp.shape
        else:
            temp.shape = (1,) + temp.shape
    return temp


_namefromtype = {'c' : 'character',
                 '1' : 'signed char',
                 'b' : 'unsigned char',
                 's' : 'short',
                 'i' : 'integer',
                 'l' : 'long integer',
                 'f' : 'float',
                 'd' : 'double',
                 'F' : 'complex float',
                 'D' : 'complex double',
                 'O' : 'object'
                 }

def typename(char):
    """Return an english name for the given typecode character.
    """
    return _namefromtype[char]

def who(vardict=None):
    """Print the Numeric arrays in the given dictionary (or globals() if None).
    """
    if vardict is None:
        print "Pass in a dictionary:  who(globals())"
        return
    sta = []
    cache = {}
    for name in vardict.keys():
        if isinstance(vardict[name],Numeric.ArrayType):
            var = vardict[name]
            idv = id(var)
            if idv in cache.keys():
                namestr = name + " (%s)" % cache[idv]
                original=0
            else:
                cache[idv] = name
                namestr = name
                original=1
            shapestr = " x ".join(map(str, var.shape))
            bytestr = str(var.itemsize()*Numeric.product(var.shape))
            sta.append([namestr, shapestr, bytestr, _namefromtype[var.typecode()], original])

    maxname = 0
    maxshape = 0
    maxbyte = 0
    totalbytes = 0
    for k in range(len(sta)):
        val = sta[k]
        if maxname < len(val[0]):
            maxname = len(val[0])
        if maxshape < len(val[1]):
            maxshape = len(val[1])
        if maxbyte < len(val[2]):
            maxbyte = len(val[2])
        if val[4]:
            totalbytes += int(val[2])

    max = Numeric.maximum
    if len(sta) > 0:
        sp1 = max(10,maxname)
        sp2 = max(10,maxshape)
        sp3 = max(10,maxbyte)
        prval = "Name %s Shape %s Bytes %s Type" % (sp1*' ', sp2*' ', sp3*' ')
        print prval + "\n" + "="*(len(prval)+5) + "\n"
        
    for k in range(len(sta)):
        val = sta[k]
        print "%s %s %s %s %s %s %s" % (val[0], ' '*(sp1-len(val[0])+4),
                                        val[1], ' '*(sp2-len(val[1])+5),
                                        val[2], ' '*(sp3-len(val[2])+5),
                                        val[3])
    print "\nUpper bound on total bytes  =       %d" % totalbytes
    return
    
def objsave(file, allglobals, *args):
    """Pickle the part of a dictionary containing the argument list
    into file string.

    Syntax:  objsave(file, globals(), obj1, obj2, ... )
    """
    fid = open(file,'w')
    savedict = {}
    for key in allglobals.keys():
        inarglist = 0
        for obj in args:
            if allglobals[key] is obj:
                inarglist = 1
                break
        if inarglist:
            savedict[key] = obj
    cPickle.dump(savedict,fid,1)
    fid.close()
        
def objload(file, allglobals):
    """Load a previously pickled dictionary and insert into given dictionary.

    Syntax:  objload(file, globals())
    """
    fid = open(file,'r')
    savedict = cPickle.load(fid)
    allglobals.update(savedict)
    fid.close()

def isscalar(num):
    if isinstance(num, ArrayType):
        return len(num.shape) == 0 and num.typecode() != 'O'
    return type(num) in ScalarType

def all_mat(args):
    return map(Matrix.Matrix,args)

    
# Selector function

def select(condlist, choicelist, default=0):
    """Returns an array comprised from different elements of choicelist
    depending on the list of conditions.

    condlist is a list of condition arrays containing ones or zeros

    choicelist is a list of choice matrices (of the "same" size as the
    arrays in condlist).  The result array has the "same" size as the
    arrays in choicelist.  If condlist is [c0,...,cN-1] then choicelist
    must be of length N.  The elements of the choicelist can then be
    represented as [v0,...,vN-1]. The default choice if none of the
    conditions are met is given as the default argument. 

    The conditions are tested in order and the first one statisfied is
    used to select the choice. In other words, the elements of the
    output array are found from the following tree (notice the order of
    the conditions matters):

    if c0: v0
    elif c1: v1
    elif c2: v2
    ...
    elif cN-1: vN-1
    else: default

    Note, that one of the condition arrays must be large enough to handle
    the largest array in the choice list.
    """
    n = len(condlist)
    n2 = len(choicelist)
    if n2 != n:
        raise ValueError, "List of cases, must be same length as the list of conditions."
    choicelist.insert(0,default)    
    S = 0
    pfac = 1
    for k in range(1,n+1):
        S += k * pfac * asarray(condlist[k-1])
        if k < n:
            pfac *= (1-asarray(condlist[k-1]))
    # handle special case of a 1-element condition but
    #  a multi-element choice
    if type(S) in ScalarType or max(asarray(S).shape)==1:
        pfac = asarray(1)
        for k in range(n2+1):
            pfac = pfac + asarray(choicelist[k])            
        S = S*ones(asarray(pfac).shape)
    return choose(S, tuple(choicelist))

def atleast_1d(tup):
    """ Force a sequence of arrays to each be at least 1D.

         Description:
            Force a sequence of arrays to each be at least 1D.  If an array
            in the sequence is 0D, the array is converted to a single
            row of values.  Otherwise, the array is unaltered.
         Arguments:
            tup -- sequence of arrays.
         Returns:
            tuple containing input arrays converted to at least 1D arrays.
    """
    new_tup = []
    for ary in tup:
        if len(ary.shape) == 0: new_tup.append(Numeric.array([ary[0]]))
        else:                   new_tup.append(ary)
    return tuple(new_tup)

def atleast_2d(tup):
    """ Force a sequence of arrays to each be at least 2D.

         Description:
            Force a sequence of arrays to each be at least 2D.  If an array
            in the sequence is 0D or 1D, the array is converted to a single
            row of values.  Otherwise, the array is unaltered.
         Arguments:
            tup -- sequence of arrays.
         Returns:
            tuple containing input arrays converted to at least 2D arrays.
    """
    new_tup = []
    for ary in tup:
        if len(ary.shape) == 0: ary = Numeric.array([ary[0]])
        if len(ary.shape) == 1: new_tup.append(ary[NewAxis,:])
        else: new_tup.append(ary)
    return tuple(new_tup)

def atleast_3d(tup):
    """ Force a sequence of arrays to each be at least 3D.

         Description:
            Force a sequence of arrays to each be at least 3D.  If an array
            in the sequence is 0D or 1D, the array is converted to a single
            1xNx1 array of values where N is the orginal length of the array.
            If the array is 2D, the array is converted to a single MxNx1
            array of values where MxN is the orginal shape of the array.
            Otherwise, the array is unaltered.
         Arguments:
            tup -- sequence of arrays.
         Returns:
            tuple containing input arrays converted to at least 2D arrays.
    """
    new_tup = []
    for ary in tup:
        if len(ary.shape) == 0: ary = Numeric.array([ary[0]])
        if len(ary.shape) == 1: new_tup.append(ary[NewAxis,:,NewAxis])
        elif len(ary.shape) == 2: new_tup.append(ary[:,:,NewAxis])
        else: new_tup.append(ary)
    return tuple(new_tup)

def vstack(tup):
    """ Stack arrays in sequence vertically (row wise)

        Description:
            Take a sequence of arrays and stack them veritcally
            to make a single array.  All arrays in the sequence
            must have the same shape along all but the first axis. 
            vstack will rebuild arrays divided by vsplit.
        Arguments:
            tup -- sequence of arrays.  All arrays must have the same 
                   shape.
        Examples:
            >>> import scipy
            >>> a = array((1,2,3))
            >>> b = array((2,3,4))
            >>> scipy.vstack((a,b))
            array([[1, 2, 3],
                   [2, 3, 4]])
            >>> a = array([[1],[2],[3]])
            >>> b = array([[2],[3],[4]])
            >>> scipy.vstack((a,b))
            array([[1],
                   [2],
                   [3],
                   [2],
                   [3],
                   [4]])

    """
    return Numeric.concatenate(atleast_2d(tup),0)

def hstack(tup):
    """ Stack arrays in sequence horizontally (column wise)

        Description:
            Take a sequence of arrays and stack them horizontally
            to make a single array.  All arrays in the sequence
            must have the same shape along all but the second axis.
            hstack will rebuild arrays divided by hsplit.
        Arguments:
            tup -- sequence of arrays.  All arrays must have the same 
                   shape.
        Examples:
            >>> import scipy
            >>> a = array((1,2,3))
            >>> b = array((2,3,4))
            >>> scipy.hstack((a,b))
            array([1, 2, 3, 2, 3, 4])
            >>> a = array([[1],[2],[3]])
            >>> b = array([[2],[3],[4]])
            >>> scipy.hstack((a,b))
            array([[1, 2],
                   [2, 3],
                   [3, 4]])

    """
    return Numeric.concatenate(atleast_1d(tup),1)

def column_stack(tup):
    """ Stack 1D arrays as columns into a 2D array

        Description:
            Take a sequence of 1D arrays and stack them as columns
            to make a single 2D array.  All arrays in the sequence
            must have the same length.
        Arguments:
            tup -- sequence of 1D arrays.  All arrays must have the same 
                   length.
        Examples:
            >>> import scipy
            >>> a = array((1,2,3))
            >>> b = array((2,3,4))
            >>> scipy.vstack((a,b))
            array([[1, 2],
                   [2, 3],
                   [3, 4]])

    """
    arrays = map(Numeric.transpose,atleast_2d(tup))
    return Numeric.concatenate(arrays,1)
    
def dstack(tup):
    """ Stack arrays in sequence depth wise (along third dimension)

        Description:
            Take a sequence of arrays and stack them along the third axis.
            All arrays in the sequence must have the same shape along all 
            but the third axis.  This is a simple way to stack 2D arrays 
            (images) into a single 3D array for processing.
            dstack will rebuild arrays divided by dsplit.
        Arguments:
            tup -- sequence of arrays.  All arrays must have the same 
                   shape.
        Examples:
            >>> import scipy
            >>> a = array((1,2,3))
            >>> b = array((2,3,4))
            >>> scipy.dstack((a,b))
            array([       [[1, 2],
                    [2, 3],
                    [3, 4]]])
            >>> a = array([[1],[2],[3]])
            >>> b = array([[2],[3],[4]])
            >>> scipy.dstack((a,b))
            array([[        [1, 2]],
                   [        [2, 3]],
                   [        [3, 4]]])
    """
    return Numeric.concatenate(atleast_3d(tup),2)

def replace_zero_by_x_arrays(sub_arys):
    for i in range(len(sub_arys)):
        if len(Numeric.shape(sub_arys[i])) == 0:
            sub_arys[i] = Numeric.array([])
        elif Numeric.sometrue(Numeric.equal(Numeric.shape(sub_arys[i]),0)):
            sub_arys[i] = Numeric.array([])   
    return sub_arys
    
def array_split(ary,indices_or_sections,axis = 0):
    """ Divide an array into a list of sub-arrays.

        Description:
           Divide ary into a list of sub-arrays along the
           specified axis.  If indices_or_sections is an integer,
           ary is divided into that many equally sized arrays.
           If it is impossible to make an equal split, each of the
           leading arrays in the list have one additional member.  If
           indices_or_sections is a list of sorted integers, its
           entries define the indexes where ary is split.

        Arguments:
           ary -- N-D array.
              Array to be divided into sub-arrays.
           indices_or_sections -- integer or 1D array.
              If integer, defines the number of (close to) equal sized
              sub-arrays.  If it is a 1D array of sorted indices, it
              defines the indexes at which ary is divided.  Any empty
              list results in a single sub-array equal to the original
              array.
           axis -- integer. default=0.
              Specifies the axis along which to split ary.
        Caveats:
           Currently, the default for axis is 0.  This
           means a 2D array is divided into multiple groups
           of rows.  This seems like the appropriate default, but
           we've agreed most other functions should default to
           axis=-1.  Perhaps we should use axis=-1 for consistency.
           However, we could also make the argument that SciPy
           works on "rows" by default.  sum() sums up rows of
           values.  split() will split data into rows.  Opinions?
    """
    try:
        Ntotal = ary.shape[axis]
    except AttributeError:
        Ntotal = len(ary)
    try: # handle scalar case.
        Nsections = len(indices_or_sections) + 1
        div_points = [0] + list(indices_or_sections) + [Ntotal]
    except TypeError: #indices_or_sections is a scalar, not an array.
        Nsections = int(indices_or_sections)
        if Nsections <= 0:
            raise ValueError, 'number sections must be larger than 0.'
        Neach_section,extras = divmod(Ntotal,Nsections)
        section_sizes = [0] + \
                        extras * [Neach_section+1] + \
                        (Nsections-extras) * [Neach_section]
        div_points = Numeric.add.accumulate(Numeric.array(section_sizes))

    sub_arys = []
    sary = Numeric.swapaxes(ary,axis,0)
    for i in range(Nsections):
        st = div_points[i]; end = div_points[i+1]
        sub_arys.append(Numeric.swapaxes(sary[st:end],axis,0))

    # there is a wierd issue with array slicing that allows
    # 0x10 arrays and other such things.  The following cluge is needed
    # to get around this issue.
    sub_arys = replace_zero_by_x_arrays(sub_arys)
    # end cluge.

    return sub_arys

def split(ary,indices_or_sections,axis=0):
    """ Divide an array into a list of sub-arrays.

        Description:
           Divide ary into a list of sub-arrays along the
           specified axis.  If indices_or_sections is an integer,
           ary is divided into that many equally sized arrays.
           If it is impossible to make an equal split, an error is 
           raised.  This is the only way this function differs from
           the array_split() function. If indices_or_sections is a 
           list of sorted integers, its entries define the indexes
           where ary is split.

        Arguments:
           ary -- N-D array.
              Array to be divided into sub-arrays.
           indices_or_sections -- integer or 1D array.
              If integer, defines the number of (close to) equal sized
              sub-arrays.  If it is a 1D array of sorted indices, it
              defines the indexes at which ary is divided.  Any empty
              list results in a single sub-array equal to the original
              array.
           axis -- integer. default=0.
              Specifies the axis along which to split ary.
        Caveats:
           Currently, the default for axis is 0.  This
           means a 2D array is divided into multiple groups
           of rows.  This seems like the appropriate default, but
           we've agreed most other functions should default to
           axis=-1.  Perhaps we should use axis=-1 for consistency.
           However, we could also make the argument that SciPy
           works on "rows" by default.  sum() sums up rows of
           values.  split() will split data into rows.  Opinions?
    """
    try: len(indices_or_sections)
    except TypeError:
        sections = indices_or_sections
        N = ary.shape[axis]
        if N % sections:
            raise ValueError, 'array split does not result in an equal division'
    res = array_split(ary,indices_or_sections,axis)
    return res

def hsplit(ary,indices_or_sections):
    """ Split ary into multiple columns of sub-arrays

        Description:
            Split a single array into multiple sub arrays.  The array is
            divided into groups of columns.  If indices_or_sections is
            an integer, ary is divided into that many equally sized sub arrays.
            If it is impossible to make the sub-arrays equally sized, the
            operation throws a ValueError exception. See array_split and
            split for other options on indices_or_sections.                        
        Arguments:
           ary -- N-D array.
              Array to be divided into sub-arrays.
           indices_or_sections -- integer or 1D array.
              If integer, defines the number of (close to) equal sized
              sub-arrays.  If it is a 1D array of sorted indices, it
              defines the indexes at which ary is divided.  Any empty
              list results in a single sub-array equal to the original
              array.
        Returns:
            sequence of sub-arrays.  The returned arrays have the same 
            number of dimensions as the input array.
        Related:
            hstack, split, array_split, vsplit, dsplit.           
        Examples:
            >>> import scipy
            >>> a= array((1,2,3,4))
            >>> scipy.hsplit(a,2)
            [array([1, 2]), array([3, 4])]
            >>> a = array([[1,2,3,4],[1,2,3,4]])
            [array([[1, 2],
                   [1, 2]]), array([[3, 4],
                   [3, 4]])]
                   
    """
    if len(Numeric.shape(ary)) == 0:
        raise ValueError, 'hsplit only works on arrays of 1 or more dimensions'
    if len(ary.shape) > 1:
        return split(ary,indices_or_sections,1)
    else:
        return split(ary,indices_or_sections,0)
        
def vsplit(ary,indices_or_sections):
    """ Split ary into multiple rows of sub-arrays

        Description:
            Split a single array into multiple sub arrays.  The array is
            divided into groups of rows.  If indices_or_sections is
            an integer, ary is divided into that many equally sized sub arrays.
            If it is impossible to make the sub-arrays equally sized, the
            operation throws a ValueError exception. See array_split and
            split for other options on indices_or_sections.
        Arguments:
           ary -- N-D array.
              Array to be divided into sub-arrays.
           indices_or_sections -- integer or 1D array.
              If integer, defines the number of (close to) equal sized
              sub-arrays.  If it is a 1D array of sorted indices, it
              defines the indexes at which ary is divided.  Any empty
              list results in a single sub-array equal to the original
              array.
        Returns:
            sequence of sub-arrays.  The returned arrays have the same 
            number of dimensions as the input array.      
        Caveats:
           How should we handle 1D arrays here?  I am currently raising
           an error when I encounter them.  Any better approach?      
           
           Should we reduce the returned array to their minium dimensions
           by getting rid of any dimensions that are 1?
        Related:
            vstack, split, array_split, hsplit, dsplit.
        Examples:
            import scipy
            >>> a = array([[1,2,3,4],
            ...            [1,2,3,4]])
            >>> scipy.vsplit(a)
            [array([       [1, 2, 3, 4]]), array([       [1, 2, 3, 4]])]
                   
    """
    if len(Numeric.shape(ary)) < 2:
        raise ValueError, 'vsplit only works on arrays of 2 or more dimensions'
    return split(ary,indices_or_sections,0)

def dsplit(ary,indices_or_sections):
    """ Split ary into multiple sub-arrays along the 3rd axis (depth)

        Description:
            Split a single array into multiple sub arrays.  The array is
            divided into groups along the 3rd axis.  If indices_or_sections is
            an integer, ary is divided into that many equally sized sub arrays.
            If it is impossible to make the sub-arrays equally sized, the
            operation throws a ValueError exception. See array_split and
            split for other options on indices_or_sections.                        
        Arguments:
           ary -- N-D array.
              Array to be divided into sub-arrays.
           indices_or_sections -- integer or 1D array.
              If integer, defines the number of (close to) equal sized
              sub-arrays.  If it is a 1D array of sorted indices, it
              defines the indexes at which ary is divided.  Any empty
              list results in a single sub-array equal to the original
              array.
        Returns:
            sequence of sub-arrays.  The returned arrays have the same 
            number of dimensions as the input array.
        Caveats:
           See vsplit caveats.       
        Related:
            dstack, split, array_split, hsplit, vsplit.
        Examples:
            >>> a = array([[[1,2,3,4],[1,2,3,4]]])
            [array([       [[1, 2],
                    [1, 2]]]), array([       [[3, 4],
                    [3, 4]]])]
                                       
    """
    if len(Numeric.shape(ary)) < 3:
        raise ValueError, 'vsplit only works on arrays of 3 or more dimensions'
    return split(ary,indices_or_sections,2)

    
# note: Got rid of keyed_split stuff here.  Perhaps revisit this in the future.

#determine the "minimum common type code" for a group of arrays.
array_kind = {'i':0, 'l': 0, 'f': 0, 'd': 0, 'F': 1, 'D': 1}
array_precision = {'i': 1, 'l': 1, 'f': 0, 'd': 1, 'F': 0, 'D': 1}
array_type = [['f', 'd'], ['F', 'D']]
def common_type(*arrays):
    kind = 0
    precision = 0
    for a in arrays:
        t = a.typecode()
        kind = max(kind, array_kind[t])
        precision = max(precision, array_precision[t])
    return array_type[kind][precision]

def trim_zeros(filt,trim='fb'):
    """Trim the leading and trailing zeros from a 1D array.
    
    Example:
        >>> import scipy
        >>> a = array((0,0,0,1,2,3,2,1,0))
        >>> scipy.trim_zeros(a)
        array([1, 2, 3, 2, 1])
    
    """
    first = 0
    if 'f' in trim or 'F' in trim:
        for i in filt:
            if i != 0.: break
            else: first = first + 1
    last = len(filt)
    if 'b' in trim or 'B' in trim:
        for i in filt[::-1]:
            if i != 0.: break
            else: last = last - 1
    return filt[first:last]

    
##def test(level=10):
##    from scipy_base.testing import module_test
##    module_test(__name__,__file__,level=level)

##def test_suite(level=1):
##    from scipy_base.testing import module_test_suite
##    return module_test_suite(__name__,__file__,level=level)




