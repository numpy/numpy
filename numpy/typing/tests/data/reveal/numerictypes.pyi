import numpy as np

reveal_type(np.cast[int])  # E: _CastFunc
reveal_type(np.cast["i8"])  # E: _CastFunc
reveal_type(np.cast[np.int64])  # E: _CastFunc

reveal_type(np.ScalarType)  # E: tuple
reveal_type(np.ScalarType[0])  # E: Type[builtins.int]
reveal_type(np.ScalarType[3])  # E: Type[builtins.bool]
reveal_type(np.ScalarType[8])  # E: Type[{csingle}]
reveal_type(np.ScalarType[10])  # E: Type[{clongdouble}]

reveal_type(np.typecodes["Character"])  # E: Literal['c']
reveal_type(np.typecodes["Complex"])  # E: Literal['FDG']
reveal_type(np.typecodes["All"])  # E: Literal['?bhilqpBHILQPefdgFDGSUVOMm']
