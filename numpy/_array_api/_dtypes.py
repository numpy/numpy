import numpy as np

# Note: we use dtype objects instead of dtype classes. The spec does not
# require any behavior on dtypes other than equality.
int8 = np.dtype('int8')
int16 = np.dtype('int16')
int32 = np.dtype('int32')
int64 = np.dtype('int64')
uint8 = np.dtype('uint8')
uint16 = np.dtype('uint16')
uint32 = np.dtype('uint32')
uint64 = np.dtype('uint64')
float32 = np.dtype('float32')
float64 = np.dtype('float64')
# Note: This name is changed
bool = np.dtype('bool')

_all_dtypes = [int8, int16, int32, int64, uint8, uint16, uint32, uint64,
               float32, float64, bool]
_boolean_dtypes = [bool]
_floating_dtypes = [float32, float64]
_integer_dtypes = [int8, int16, int32, int64, uint8, uint16, uint32, uint64]
_integer_or_boolean_dtypes = [bool, int8, int16, int32, int64, uint8, uint16, uint32, uint64]
_numeric_dtypes = [float32, float64, int8, int16, int32, int64, uint8, uint16, uint32, uint64]
