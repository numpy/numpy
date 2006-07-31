
from lib import add_newdoc

add_newdoc('numpy.core','dtype',
           [('fields', "Fields of the data-type or None if no fields"),
            ('names', "Names of fields or None if no fields"),
            ('alignment', "Needed alignment for this data-type"),
            ('byteorder',
             "Little-endian (<), big-endian (>), native (=), or "\
             "not-applicable (|)"),
            ('char', "Letter typecode for this data-type"),
            ('type', "Type object associated with this data-type"),
            ('kind', "Character giving type-family of this data-type"),
            ('itemsize', "Size of each item"),
            ('hasobject', "Non-zero if Python objects are in "\
             "this data-type"),
            ('num', "Internally-used number for builtin base")
            ]
           )

add_newdoc('numpy.core.multiarray','array',
"""array(object, dtype=None, copy=1,order=None, subok=0,ndmin=0)
              
Return an array from object with the specified date-type.

Inputs:
  object - an array, any object exposing the array interface, any 
            object whose __array__ method returns an array, or any 
            (nested) sequence.
  dtype  - The desired data-type for the array.  If not given, then
            the type will be determined as the minimum type required
            to hold the objects in the sequence.  This argument can only
            be used to 'upcast' the array.  For downcasting, use the 
            .astype(t) method.
  copy   - If true, then force a copy.  Otherwise a copy will only occur
            if __array__ returns a copy, obj is a nested sequence, or 
            a copy is needed to satisfy any of the other requirements
  order  - Specify the order of the array.  If order is 'C', then the
            array will be in C-contiguous order (last-index varies the
            fastest).  If order is 'FORTRAN', then the returned array
            will be in Fortran-contiguous order (first-index varies the
            fastest).  If order is None, then the returned array may
            be in either C-, or Fortran-contiguous order or even
            discontiguous.
  subok  - If True, then sub-classes will be passed-through, otherwise
            the returned array will be forced to be a base-class array
  ndmin  - Specifies the minimum number of dimensions that the resulting
            array should have.  1's will be pre-pended to the shape as
            needed to meet this requirement.

""")

add_newdoc('numpy.core.multiarray','empty',
"""empty((d1,...,dn),dtype=float,order='C')

Return a new array of shape (d1,...,dn) and given type with all its
entries uninitialized. This can be faster than zeros.

""")


add_newdoc('numpy.core.multiarray','scalar',
"""scalar(dtype,obj)

Return a new scalar array of the given type initialized with
obj. Mainly for pickle support.  The dtype must be a valid data-type
descriptor.  If dtype corresponds to an OBJECT descriptor, then obj
can be any object, otherwise obj must be a string. If obj is not given
it will be interpreted as None for object type and zeros for all other
types.

""")

add_newdoc('numpy.core.multiarray','zeros',
"""zeros((d1,...,dn),dtype=float,order='C')

Return a new array of shape (d1,...,dn) and type typecode with all
it's entries initialized to zero.

""")

add_newdoc('numpy.core.multiarray','set_typeDict',
"""set_typeDict(dict)

Set the internal dictionary that can look up an array type using a
registered code.

""")

add_newdoc('numpy.core.multiarray','fromstring',
"""fromstring(string, dtype=float, count=-1, sep='')

Return a new 1d array initialized from the raw binary data in string.

If count is positive, the new array will have count elements, otherwise its
size is determined by the size of string.  If sep is not empty then the
string is interpreted in ASCII mode and converted to the desired number type
using sep as the separator between elements (extra whitespace is ignored).

""")

add_newdoc('numpy.core.multiarray','fromstring',
"""fromiter(iterable, dtype, count=-1)

Return a new 1d array initialized from iterable. If count is
nonegative, the new array will have count elements, otherwise it's
size is determined by the generator.

""")

add_newdoc('numpy.core.multiarray','fromfile',
"""fromfile(file=, dtype=float, count=-1, sep='')

Return an array of the given data type from a (text or binary) file.
The file argument can be an open file or a string with the name of a
file to read from.  If count==-1, then the entire file is read,
otherwise count is the number of items of the given type read in.  If
sep is '' then read a binary file, otherwise it gives the separator
between elements in a text file.

WARNING: This function should be used sparingly, as it is not a
platform-independent method of persistence.  But it can be useful to
read in simply-formatted or binary data quickly.

""")

add_newdoc('numpy.core.multiarray','frombuffer',
"""frombuffer(buffer=, dtype=float, count=-1, offset=0)

Returns a 1-d array of data type dtype from buffer. The buffer
argument must be an object that exposes the buffer interface.  If
count is -1 then the entire buffer is used, otherwise, count is the
size of the output.  If offset is given then jump that far into the
buffer. If the buffer has data that is out not in machine byte-order,
than use a propert data type descriptor. The data will not be
byteswapped, but the array will manage it in future operations.

""")

add_newdoc('numpy.core.multiarray','concatenate',
"""concatenate((a1, a2, ...), axis=None)

Join arrays together.

The tuple of sequences (a1, a2, ...) are joined along the given axis
(default is the first one) into a single numpy array.
    
Example:
    
>>> concatenate( ([0,1,2], [5,6,7]) )
array([0, 1, 2, 5, 6, 7])

""")

add_newdoc('numpy.core.multiarray','inner',
"""inner(a,b)

Returns the dot product of two arrays, which has shape a.shape[:-1] +
b.shape[:-1] with elements computed by the product of the elements
from the last dimensions of a and b.

""")

add_newdoc('numpy.core','fastCopyAndTranspose',
"""_fastCopyAndTranspose(a)""")

add_newdoc('numpy.core.multiarray','correlate',
"""cross_correlate(a,v, mode=0)""")

add_newdoc('numpy.core.multiarray','arange',
"""arange([start,] stop[, step,], dtype=None)

For integer arguments, just like range() except it returns an array
whose type can be specified by the keyword argument dtype.  If dtype
is not specified, the type of the result is deduced from the type of
the arguments.

For floating point arguments, the length of the result is ceil((stop -
start)/step).  This rule may result in the last element of the result
being greater than stop.

""")

add_newdoc('numpy.core.multiarray','_get_ndarray_c_version',
"""_get_ndarray_c_version()

Return the compile time NDARRAY_VERSION number.

""")

add_newdoc('numpy.core.multiarray','_reconstruct',
"""_reconstruct(subtype, shape, dtype)

Construct an empty array. Used by Pickles.

""")


add_newdoc('numpy.core.multiarray','set_string_function',
"""set_string_function(f, repr=1)

Set the python function f to be the function used to obtain a pretty
printable string version of an array whenever an array is printed.
f(M) should expect an array argument M, and should return a string
consisting of the desired representation of M for printing.

""")

add_newdoc('numpy.core.multiarray','set_numeric_ops',
"""set_numeric_ops(op=func, ...)

Set some or all of the number methods for all array objects.  Don't
forget **dict can be used as the argument list.  Return the functions
that were replaced, which can be stored and set later.

""")

add_newdoc('numpy.core.multiarray','where',
"""where(condition, | x, y)

The result is shaped like condition and has elements of x and y where
condition is respectively true or false.  If x or y are not given,
then it is equivalent to condition.nonzero().

To group the indices by element, rather than dimension, use

    transpose(where(condition, | x, y))
    
instead. This always results in a 2d array, with a row of indices for
each element that satisfies the condition.

""")


add_newdoc('numpy.core.multiarray','lexsort',
"""lexsort(keys=, axis=-1)

Return an array of indices similar to argsort, except the sorting is
done using the provided sorting keys.  First the sort is done using
key[0], then the resulting list of indices is further manipulated by
sorting on key[1], and so forth. The result is a sort on multiple
keys.  If the keys represented columns of a spreadsheet, for example,
this would sort using multiple columns.  The keys argument must be a
sequence of things that can be converted to arrays of the same shape.

""")

add_newdoc('numpy.core.multiarray','can_cast',
"""can_cast(from=d1, to=d2)

Returns True if data type d1 can be cast to data type d2 without
losing precision.

""")

add_newdoc('numpy.core.multiarray','newbuffer',
"""newbuffer(size)

Return a new uninitialized buffer object of size bytes

""")

add_newdoc('numpy.core.multiarray','getbuffer',
"""getbuffer(obj [,offset[, size]])

Create a buffer object from the given object referencing a slice of
length size starting at offset.  Default is the entire buffer. A
read-write buffer is attempted followed by a read-only buffer.

""")
