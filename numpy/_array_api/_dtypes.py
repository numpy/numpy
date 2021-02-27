from .. import int8, int16, int32, int64, uint8, uint16, uint32, uint64, float32, float64
# Note: This name is changed
from .. import bool_ as bool

_all_dtypes = [int8, int16, int32, int64, uint8, uint16, uint32, uint64,
               float32, float64, bool]
_boolean_dtypes = [bool]
_floating_dtypes = [float32, float64]
_integer_dtypes = [int8, int16, int32, int64, uint8, uint16, uint32, uint64]
_integer_or_boolean_dtypes = [bool, int8, int16, int32, int64, uint8, uint16, uint32, uint64]
_numeric_dtypes = [float32, float64, int8, int16, int32, int64, uint8, uint16, uint32, uint64]
