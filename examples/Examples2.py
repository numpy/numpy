#Exaples of ctypes functions
import numpy as np
import ctypes

x = np.ctypeslib.as_array([[0,1] , [2,3]])
print(x)
print(x.shape)
print(x.strides)
print(x[0].ctypes.data)
print(x[1].ctypes.data)

y = np.full(6, 69, dtype=int)
y = np.ctypeslib.as_ctypes(y)
y._type_

print(x[1][1] * y[0])

x = np.ctypeslib.as_array(y, int)
print(x)
print(x.shape)
print(x.strides)
print(x.ctypes.data)

print('x =',x[:],'\n'
      + 'y =',y[:])

y = np.full(6, 6.66, dtype=float)
y = np.ctypeslib.as_ctypes(y)
print(y,'\n',y[:])

#There is the Issue 6176
y = np.full(6, True, dtype=bool)
y = np.ctypeslib.as_ctypes(y)
print(y)
