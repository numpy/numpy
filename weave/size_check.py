from Numeric import *

# The following try/except so that non-SciPy users can still use blitz
try:
    from scipy_base.fastumath import *
except:
    pass # scipy_base.fastumath not available    

from ast_tools import *
from types import *
import sys

def time_it():
    import time
    
    expr = "ex[:,1:,1:] =   ca_x[:,1:,1:] * ex[:,1:,1:]" \
                         "+ cb_y_x[:,1:,1:] * (hz[:,1:,1:] - hz[:,:-1,1:])" \
                         "- cb_z_x[:,1:,1:] * (hy[:,1:,1:] - hy[:,1:,:-1])"        
    ex = ones((10,10,10),typecode=Float32)
    ca_x = ones((10,10,10),typecode=Float32)
    cb_y_x = ones((10,10,10),typecode=Float32)
    cb_z_x = ones((10,10,10),typecode=Float32)
    hz = ones((10,10,10),typecode=Float32)
    hy = ones((10,10,10),typecode=Float32)
    
    N = 1
    t1 = time.time()
    for i in range(N):
        passed = check_expr(expr,locals())
    t2 = time.time()
    print 'time per call:', (t2 - t1)/N
    print 'passed:', passed
    
def check_expr(expr,local_vars,global_vars={}):
    """ Currently only checks expressions (not suites).
        Doesn't check that lhs = rhs. checked by compiled func though
    """
    values ={}
    
    #first handle the globals
    for var,val in global_vars.items():
        if type(val) in [ArrayType]: 
            values[var] = dummy_array(val,name=var)
        elif type(val) in [IntType,LongType,FloatType,ComplexType]:    
            values[var] = val
    #now handle the locals        
    for var,val in local_vars.items():
        if type(val) in [ArrayType]: 
            values[var] = dummy_array(val,name=var)
        elif type(val) in [IntType,LongType,FloatType,ComplexType]:    
            values[var] = val
    exec(expr,values)
    try:
        exec(expr,values)
    except:
        try:
            eval(expr,values)
        except:
            return 0    
    return 1                
        
empty = array(())
empty_slice = slice(None)

def make_same_length(x,y):
    try:
        Nx = len(x)
    except:
        Nx = 0
    try:
        Ny = len(y)
    except:
        Ny = 0
    if Nx == Ny == 0:
        return empty,empty
    elif Nx == Ny:
        return asarray(x),asarray(y)
    else:    
        diff = abs(Nx - Ny)
        front = ones(diff,Int)
        if Nx > Ny:
            return asarray(x), concatenate((front,y))
        elif Ny > Nx:
            return concatenate((front,x)),asarray(y)            

def binary_op_size(xx,yy):
    """ This returns the resulting size from operating on xx, and yy
        with a binary operator.  It accounts for broadcasting, and
        throws errors if the array sizes are incompatible.
    """
    x,y = make_same_length(xx,yy)
    res = zeros(len(x))
    for i in range(len(x)):
        if x[i] == y[i]:
            res[i] = x[i]
        elif x[i] == 1:
            res[i] = y[i]
        elif y[i] == 1:
            res[i] = x[i]
        else:
            # offer more information here about which variables.
            raise ValueError, "frames are not aligned"
    return res    
class dummy_array:
    def __init__(self,ary,ary_is_shape = 0,name=None):
        self.name = name
        if ary_is_shape:
            self.shape = ary
            #self.shape = asarray(ary)
        else:
            try:
                self.shape = shape(ary)
            except:
                self.shape = empty
        #self.value = ary        
    def binary_op(self,other):
        try: 
            x = other.shape
        except AttributeError:
            x = empty                      
        new_shape = binary_op_size(self.shape,x)
        return dummy_array(new_shape,1)
    def __cmp__(self,other):
        # This isn't an exact compare, but does work for == 
        # cluge for Numeric
        if type(other) in [IntType,LongType,FloatType,ComplexType]:
            return 0
        if len(self.shape) == len(other.shape) == 0:
            return 0
        return not alltrue(equal(self.shape,other.shape))

    def __add__(self,other): return self.binary_op(other)
    def __radd__(self,other): return self.binary_op(other)
    def __sub__(self,other): return self.binary_op(other)
    def __rsub__(self,other): return self.binary_op(other)
    def __mul__(self,other): return self.binary_op(other)
    def __rmul__(self,other): return self.binary_op(other)
    def __div__(self,other): return self.binary_op(other)
    def __rdiv__(self,other): return self.binary_op(other)
    def __mod__(self,other): return self.binary_op(other)
    def __rmod__(self,other): return self.binary_op(other)
    def __lshift__(self,other): return self.binary_op(other)
    def __rshift__(self,other): return self.binary_op(other)
    # unary ops
    def __neg__(self,other): return self
    def __pos__(self,other): return self
    def __abs__(self,other): return self
    def __invert__(self,other): return self   
    # Not sure what to do with coersion ops.  Ignore for now.
    #
    # not currently supported by compiler.
    # __divmod__
    # __pow__
    # __rpow__
    # __and__
    # __or__
    # __xor__
    # item access and slicing    
    def __setitem__(self,indices,val):
        #ignore for now
        pass
    def __len__(self):
        return self.shape[0]
    def __getslice__(self,i,j):
        # enabling the following would make class compatible with
        # lists.  Its current incarnation is compatible with arrays.
        # Both this and Numeric should have this FIXED to correspond
        # to lists.
        #i = max(i, 0); j = max(j, 0)
        return self.__getitem__((slice(i,j),))
    def __getitem__(self,indices):
        # ayeyaya this is a mess
        #print indices, type(indices), indices.shape       
        if type(indices) is not TupleType:
            indices = (indices,)
        if Ellipsis in indices:
            raise IndexError, "Ellipsis not currently supported"
        new_dims = []            
        dim = 0    
        for index in indices:
            try:
                dim_len = self.shape[dim]
            except IndexError:
                raise IndexError, "To many indices specified"
                
            #if (type(index) is SliceType and index.start == index.stop == index.step):
            if (index is empty_slice):
                slc_len = dim_len
            elif type(index) is SliceType:
                beg,end,step = index.start,index.stop,index.step                
                # handle if they are dummy arrays
                #if hasattr(beg,'value') and type(beg.value) != ArrayType:
                #    beg = beg.value
                #if hasattr(end,'value') and type(end.value) != ArrayType:
                #    end = end.value
                #if hasattr(step,'value') and type(step.value) != ArrayType:
                #    step = step.value    
                if beg is None: beg = 0
                if end == sys.maxint or  end is None:
                    end = dim_len
                if step is None: 
                    step = 1
                    
                if beg < 0: beg += dim_len
                if end < 0: end += dim_len
                # the following is list like behavior,
                # which isn't adhered to by arrays. 
                # FIX THIS ANOMOLY IN NUMERIC!
                if beg < 0: beg = 0
                if beg > dim_len: beg = dim_len
                if end < 0: end = 0
                if end > dim_len: end = dim_len
                # This is rubbish. 
                if  beg == end:
                    beg,end,step = 0,0,1
                elif  beg >= dim_len and step > 0:
                    beg,end,step = 0,0,1
                #elif index.step > 0 and beg <= end:
                elif step > 0 and beg <= end:
                    pass #slc_len = abs(divide(end-beg-1,step)+1)     
                # handle [::-1] and [-1::-1] correctly    
                #elif index.step > 0 and beg > end:
                elif step > 0 and beg > end:
                    beg,end,step = 0,0,1
                elif(step < 0 and index.start is None and index.stop is None):
                    beg,end,step = 0,dim_len,-step
                elif(step < 0 and index.start is None):
                    # +1 because negative stepping is inclusive
                    beg,end,step = end+1,dim_len,-step 
                elif(step < 0 and index.stop is None):
                    beg,end,step = 0,beg+1,-step
                elif(step < 0 and beg > end):                    
                    beg,end,step = end,beg,-step
                elif(step < 0 and beg < end):                    
                    beg,end,step = 0,0,-step
                slc_len = abs(divide(end-beg-1,step)+1)
                new_dims.append(slc_len)
            else:
                if index < 0: index += dim_len
                if index >=0 and index < dim_len:
                    #this reduces the array dimensions by one
                    pass
                else:
                    raise IndexError, "Index out of range"            
            dim += 1        
        new_dims.extend(self.shape[dim:])
        if 0 in new_dims:
            raise IndexError, "Zero length slices not currently supported"
        return dummy_array(new_dims,1)
    def __repr__(self):
        val = str((self.name, str(self.shape)))
        return val    

def unary(ary):
    return ary

def not_implemented(ary):
    return ary
    
#all imported from Numeric and need to be reassigned.
unary_op = [arccos, arcsin, arctan, cos, cosh, sin, sinh, 
            exp,ceil,floor,fabs,log,log10,sqrt]

unsupported = [argmin,argmax, argsort,around, absolute,sign,negative,floor]

for func in unary_op:
    func = unary
    
for func in unsupported:
    func = not_implemented
    
def reduction(ary,axis=0):
    if axis < 0:
        axis += len(ary.shape) 
    if axis < 0 or axis >= len(ary.shape):
        raise ValueError, "Dimension not in array"        
    new_dims = list(ary.shape[:axis]) + list(ary.shape[axis+1:])
    return dummy_array(new_dims,1)           

# functions currently not supported by compiler
# reductions are gonna take some array reordering for the general case,
# so this is gonna take some thought (probably some tree manipulation).
def take(ary,axis=0): raise NotImplemented
# and all the rest
