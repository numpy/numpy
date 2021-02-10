from typing import List, Any
import numpy as np

AR_LIKE_b: List[bool]
AR_LIKE_u: List[np.uint32]
AR_LIKE_i: List[int]
AR_LIKE_f: List[float]
AR_LIKE_c: List[complex]
AR_LIKE_U: List[str]

OUT_f: np.ndarray[Any, np.dtype[np.float64]]

reveal_type(  # E: Union[numpy.bool_, numpy.ndarray[Any, numpy.dtype[numpy.bool_]]
    np.einsum("i,i->i", AR_LIKE_b, AR_LIKE_b)
)
reveal_type(  # E: Union[numpy.unsignedinteger[Any], numpy.ndarray[Any, numpy.dtype[numpy.unsignedinteger[Any]]]
    np.einsum("i,i->i", AR_LIKE_u, AR_LIKE_u)
)
reveal_type(  # E: Union[numpy.signedinteger[Any], numpy.ndarray[Any, numpy.dtype[numpy.signedinteger[Any]]]
    np.einsum("i,i->i", AR_LIKE_i, AR_LIKE_i)
)
reveal_type(  # E: Union[numpy.floating[Any], numpy.ndarray[Any, numpy.dtype[numpy.floating[Any]]]
    np.einsum("i,i->i", AR_LIKE_f, AR_LIKE_f)
)
reveal_type(  # E: Union[numpy.complexfloating[Any, Any], numpy.ndarray[Any, numpy.dtype[numpy.complexfloating[Any, Any]]]
    np.einsum("i,i->i", AR_LIKE_c, AR_LIKE_c)
)
reveal_type(  # E: Union[numpy.signedinteger[Any], numpy.ndarray[Any, numpy.dtype[numpy.signedinteger[Any]]]
    np.einsum("i,i->i", AR_LIKE_b, AR_LIKE_i)
)
reveal_type(  # E: Union[numpy.complexfloating[Any, Any], numpy.ndarray[Any, numpy.dtype[numpy.complexfloating[Any, Any]]]
    np.einsum("i,i,i,i->i", AR_LIKE_b, AR_LIKE_u, AR_LIKE_i, AR_LIKE_c)
)

reveal_type(  # E: Union[{float64}, numpy.ndarray[Any, numpy.dtype[{float64}]]
    np.einsum("i,i->i", AR_LIKE_c, AR_LIKE_c, out=OUT_f)
)
reveal_type(  # E: numpy.ndarray[Any, numpy.dtype[numpy.str_]
    np.einsum("i,i->i", AR_LIKE_U, AR_LIKE_U, dtype=bool, casting="unsafe", out=OUT_f)
)
reveal_type(  # E: Union[numpy.complexfloating[Any, Any], numpy.ndarray[Any, numpy.dtype[numpy.complexfloating[Any, Any]]]
    np.einsum("i,i->i", AR_LIKE_f, AR_LIKE_f, dtype="c16")
)
reveal_type(  # E: Any
    np.einsum("i,i->i", AR_LIKE_U, AR_LIKE_U, dtype=bool, casting="unsafe")
)

reveal_type(  # E: Tuple[builtins.list[Any], builtins.str]
    np.einsum_path("i,i->i", AR_LIKE_b, AR_LIKE_b)
)
reveal_type(  # E: Tuple[builtins.list[Any], builtins.str]
    np.einsum_path("i,i->i", AR_LIKE_u, AR_LIKE_u)
)
reveal_type(  # E: Tuple[builtins.list[Any], builtins.str]
    np.einsum_path("i,i->i", AR_LIKE_i, AR_LIKE_i)
)
reveal_type(  # E: Tuple[builtins.list[Any], builtins.str]
    np.einsum_path("i,i->i", AR_LIKE_f, AR_LIKE_f)
)
reveal_type(  # E: Tuple[builtins.list[Any], builtins.str]
    np.einsum_path("i,i->i", AR_LIKE_c, AR_LIKE_c)
)
reveal_type(  # E: Tuple[builtins.list[Any], builtins.str]
    np.einsum_path("i,i->i", AR_LIKE_b, AR_LIKE_i)
)
reveal_type(  # E: Tuple[builtins.list[Any], builtins.str]
    np.einsum_path("i,i,i,i->i", AR_LIKE_b, AR_LIKE_u, AR_LIKE_i, AR_LIKE_c)
)
