__all__ = ['atleast_1d','atleast_2d','atleast_3d','vstack','hstack',
           'column_stack','row_stack', 'dstack','array_split','split','hsplit',
           'vsplit','dsplit','apply_over_axes','expand_dims',
           'apply_along_axis', 'tile', 'kron']

import numpy.core.numeric as _nx
from numpy.core.numeric import asarray, zeros, newaxis, outer, \
     concatenate, isscalar, array, asanyarray
from numpy.core.fromnumeric import product, reshape

def apply_along_axis(func1d,axis,arr,*args):
    """ Execute func1d(arr[i],*args) where func1d takes 1-D arrays
        and arr is an N-d array.  i varies so as to apply the function
        along the given axis for each 1-d subarray in arr.
    """
    arr = asarray(arr)
    nd = arr.ndim
    if axis < 0:
        axis += nd
    if (axis >= nd):
        raise ValueError("axis must be less than arr.ndim; axis=%d, rank=%d."
            % (axis,nd))
    ind = [0]*(nd-1)
    i = zeros(nd,'O')
    indlist = range(nd)
    indlist.remove(axis)
    i[axis] = slice(None,None)
    outshape = asarray(arr.shape).take(indlist)
    i.put(indlist, ind)
    res = func1d(arr[tuple(i.tolist())],*args)
    #  if res is a number, then we have a smaller output array
    if isscalar(res):
        outarr = zeros(outshape,asarray(res).dtype)
        outarr[ind] = res
        Ntot = product(outshape)
        k = 1
        while k < Ntot:
            # increment the index
            ind[-1] += 1
            n = -1
            while (ind[n] >= outshape[n]) and (n > (1-nd)):
                ind[n-1] += 1
                ind[n] = 0
                n -= 1
            i.put(indlist,ind)
            res = func1d(arr[tuple(i.tolist())],*args)
            outarr[ind] = res
            k += 1
        return outarr
    else:
        Ntot = product(outshape)
        holdshape = outshape
        outshape = list(arr.shape)
        outshape[axis] = len(res)
        outarr = zeros(outshape,asarray(res).dtype)
        outarr[tuple(i.tolist())] = res
        k = 1
        while k < Ntot:
            # increment the index
            ind[-1] += 1
            n = -1
            while (ind[n] >= holdshape[n]) and (n > (1-nd)):
                ind[n-1] += 1
                ind[n] = 0
                n -= 1
            i.put(indlist, ind)
            res = func1d(arr[tuple(i.tolist())],*args)
            outarr[tuple(i.tolist())] = res
            k += 1
        return outarr


def apply_over_axes(func, a, axes):
    """Apply a function repeatedly over multiple axes, keeping the same shape
    for the resulting array.

    func is called as res = func(a, axis).  The result is assumed
    to be either the same shape as a or have one less dimension.
    This call is repeated for each axis in the axes sequence.
    """
    val = asarray(a)
    N = a.ndim
    if array(axes).ndim == 0:
        axes = (axes,)
    for axis in axes:
        if axis < 0: axis = N + axis
        args = (val, axis)
        res = func(*args)
        if res.ndim == val.ndim:
            val = res
        else:
            res = expand_dims(res,axis)
            if res.ndim == val.ndim:
                val = res
            else:
                raise ValueError, "function is not returning"\
                      " an array of correct shape"
    return val

def expand_dims(a, axis):
    """Expand the shape of a by including newaxis before given axis.
    """
    a = asarray(a)
    shape = a.shape
    if axis < 0:
        axis = axis + len(shape) + 1
    return a.reshape(shape[:axis] + (1,) + shape[axis:])


def atleast_1d(*arys):
    """ Force a sequence of arrays to each be at least 1D.

         Description:
            Force an array to be at least 1D.  If an array is 0D, the
            array is converted to a single row of values.  Otherwise,
            the array is unaltered.
         Arguments:
            *arys -- arrays to be converted to 1 or more dimensional array.
         Returns:
            input array converted to at least 1D array.
    """
    res = []
    for ary in arys:
        res.append(array(ary,copy=False,subok=True,ndmin=1))
    if len(res) == 1:
        return res[0]
    else:
        return res

def atleast_2d(*arys):
    """ Force a sequence of arrays to each be at least 2D.

         Description:
            Force an array to each be at least 2D.  If the array
            is 0D or 1D, the array is converted to a single
            row of values.  Otherwise, the array is unaltered.
         Arguments:
            arys -- arrays to be converted to 2 or more dimensional array.
         Returns:
            input array converted to at least 2D array.
    """
    res = []
    for ary in arys:
        res.append(array(ary,copy=False,subok=True,ndmin=2))
    if len(res) == 1:
        return res[0]
    else:
        return res

def atleast_3d(*arys):
    """ Force a sequence of arrays to each be at least 3D.

         Description:
            Force an array each be at least 3D.  If the array is 0D or 1D,
            the array is converted to a single 1xNx1 array of values where
            N is the orginal length of the array. If the array is 2D, the
            array is converted to a single MxNx1 array of values where MxN
            is the orginal shape of the array. Otherwise, the array is
            unaltered.
         Arguments:
            arys -- arrays to be converted to 3 or more dimensional array.
         Returns:
            input array converted to at least 3D array.
    """
    res = []
    for ary in arys:
        ary = asarray(ary)
        if len(ary.shape) == 0:
            result = ary.reshape(1,1,1)
        elif len(ary.shape) == 1:
            result = ary[newaxis,:,newaxis]
        elif len(ary.shape) == 2:
            result = ary[:,:,newaxis]
        else:
            result = ary
        res.append(result)
    if len(res) == 1:
        return res[0]
    else:
        return res


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
            >>> import numpy
            >>> a = array((1,2,3))
            >>> b = array((2,3,4))
            >>> numpy.vstack((a,b))
            array([[1, 2, 3],
                   [2, 3, 4]])
            >>> a = array([[1],[2],[3]])
            >>> b = array([[2],[3],[4]])
            >>> numpy.vstack((a,b))
            array([[1],
                   [2],
                   [3],
                   [2],
                   [3],
                   [4]])

    """
    return _nx.concatenate(map(atleast_2d,tup),0)

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
            >>> import numpy
            >>> a = array((1,2,3))
            >>> b = array((2,3,4))
            >>> numpy.hstack((a,b))
            array([1, 2, 3, 2, 3, 4])
            >>> a = array([[1],[2],[3]])
            >>> b = array([[2],[3],[4]])
            >>> numpy.hstack((a,b))
            array([[1, 2],
                   [2, 3],
                   [3, 4]])

    """
    return _nx.concatenate(map(atleast_1d,tup),1)

row_stack = vstack

def column_stack(tup):
    """ Stack 1D arrays as columns into a 2D array

        Description:
            Take a sequence of 1D arrays and stack them as columns
            to make a single 2D array.  All arrays in the sequence
            must have the same first dimension.  2D arrays are
            stacked as-is, just like with hstack.  1D arrays are turned
            into 2D columns first.
            
        Arguments:
            tup -- sequence of 1D or 2D arrays.  All arrays must have the same
                   first dimension.
        Examples:
            >>> import numpy
            >>> a = array((1,2,3))
            >>> b = array((2,3,4))
            >>> numpy.column_stack((a,b))
            array([[1, 2],
                   [2, 3],
                   [3, 4]])

    """
    arrays = []
    for v in tup:
        arr = array(v,copy=False,subok=True)
        if arr.ndim < 2:
            arr = array(arr,copy=False,subok=True,ndmin=2).T
        arrays.append(arr)
    return _nx.concatenate(arrays,1)

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
            >>> import numpy
            >>> a = array((1,2,3))
            >>> b = array((2,3,4))
            >>> numpy.dstack((a,b))
            array([       [[1, 2],
                    [2, 3],
                    [3, 4]]])
            >>> a = array([[1],[2],[3]])
            >>> b = array([[2],[3],[4]])
            >>> numpy.dstack((a,b))
            array([[        [1, 2]],
                   [        [2, 3]],
                   [        [3, 4]]])
    """
    return _nx.concatenate(map(atleast_3d,tup),2)

def _replace_zero_by_x_arrays(sub_arys):
    for i in range(len(sub_arys)):
        if len(_nx.shape(sub_arys[i])) == 0:
            sub_arys[i] = _nx.array([])
        elif _nx.sometrue(_nx.equal(_nx.shape(sub_arys[i]),0)):
            sub_arys[i] = _nx.array([])
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
           of rows.  This seems like the appropriate default,
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
        div_points = _nx.array(section_sizes).cumsum()

    sub_arys = []
    sary = _nx.swapaxes(ary,axis,0)
    for i in range(Nsections):
        st = div_points[i]; end = div_points[i+1]
        sub_arys.append(_nx.swapaxes(sary[st:end],axis,0))

    # there is a wierd issue with array slicing that allows
    # 0x10 arrays and other such things.  The following cluge is needed
    # to get around this issue.
    sub_arys = _replace_zero_by_x_arrays(sub_arys)
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
           of rows.  This seems like the appropriate default
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
            >>> import numpy
            >>> a= array((1,2,3,4))
            >>> numpy.hsplit(a,2)
            [array([1, 2]), array([3, 4])]
            >>> a = array([[1,2,3,4],[1,2,3,4]])
            [array([[1, 2],
                   [1, 2]]), array([[3, 4],
                   [3, 4]])]

    """
    if len(_nx.shape(ary)) == 0:
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
            import numpy
            >>> a = array([[1,2,3,4],
            ...            [1,2,3,4]])
            >>> numpy.vsplit(a)
            [array([       [1, 2, 3, 4]]), array([       [1, 2, 3, 4]])]

    """
    if len(_nx.shape(ary)) < 2:
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
    if len(_nx.shape(ary)) < 3:
        raise ValueError, 'vsplit only works on arrays of 3 or more dimensions'
    return split(ary,indices_or_sections,2)

def _getwrapper(*args):
    """Find the wrapper for the array with the highest priority.

    In case of ties, leftmost wins. If no wrapper is found, return None
    """
    wrappers = [(getattr(x, '__array_priority__', 0), -i,
                 x.__array_wrap__) for i, x in enumerate(args) 
                                   if hasattr(x, '__array_wrap__')]
    wrappers.sort()
    if wrappers:
        return wrappers[-1][-1]
    return None 

def kron(a,b):
    """kronecker product of a and b

    Kronecker product of two matrices is block matrix
    [[ a[ 0 ,0]*b, a[ 0 ,1]*b, ... , a[ 0 ,n-1]*b  ],
     [ ...                                   ...   ],
     [ a[m-1,0]*b, a[m-1,1]*b, ... , a[m-1,n-1]*b  ]]
    """
    wrapper = _getwrapper(a, b)
    a = asanyarray(a)
    b = asanyarray(b)
    if not (len(a.shape) == len(b.shape) == 2):
        raise ValueError("a and b must both be two dimensional")
    if not a.flags.contiguous:
        a = reshape(a, a.shape)
    if not b.flags.contiguous:
        b = reshape(b, b.shape)
    o = outer(a,b)
    o=o.reshape(a.shape + b.shape)
    result = concatenate(concatenate(o, axis=1), axis=1)
    if wrapper is not None:
        result = wrapper(result)
    return result

def tile(A, reps):
    """Repeat an array the number of times given in the integer tuple, tup.

    If reps has length d, the result will have dimension of max(d, A.ndim).
    If reps is scalar it is treated as a 1-tuple.

    If A.ndim < d, A is promoted to be d-dimensional by prepending new axes.
    So a shape (3,) array is promoted to (1,3) for 2-D replication,
    or shape (1,1,3) for 3-D replication.
    If this is not the desired behavior, promote A to d-dimensions manually
    before calling this function.

    If d < A.ndim, tup is promoted to A.ndim by pre-pending 1's to it.  Thus
    for an A.shape of (2,3,4,5), a tup of (2,2) is treated as (1,1,2,2)


    Examples:
    >>> a = array([0,1,2])
    >>> tile(a,2)
    array([0, 1, 2, 0, 1, 2])
    >>> tile(a,(1,2))
    array([[0, 1, 2, 0, 1, 2]])
    >>> tile(a,(2,2))
    array([[0, 1, 2, 0, 1, 2],
           [0, 1, 2, 0, 1, 2]])
    >>> tile(a,(2,1,2))
    array([[[0, 1, 2, 0, 1, 2]],

           [[0, 1, 2, 0, 1, 2]]])

    See Also:
       repeat
    """
    try:
        tup = tuple(reps)
    except TypeError:
        tup = (reps,)
    d = len(tup)
    c = _nx.array(A,copy=False,subok=True,ndmin=d)
    shape = list(c.shape)
    n = c.size
    if (d < c.ndim):
        tup = (1,)*(c.ndim-d) + tup
    for i, nrep in enumerate(tup):
        if nrep!=1:
            c = c.reshape(-1,n).repeat(nrep,0)
        dim_in = shape[i]
        dim_out = dim_in*nrep
        shape[i] = dim_out
        n /= dim_in
    return c.reshape(shape)

