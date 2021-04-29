from typing import List, Any
import numpy as np

AR_LIKE_b: List[bool]
AR_LIKE_u: List[np.uint32]
AR_LIKE_i: List[int]
AR_LIKE_f: List[float]
AR_LIKE_c: List[complex]
AR_LIKE_U: List[str]

OUT_f: np.ndarray[Any, np.dtype[np.float64]]

reveal_type(np.einsum("i,i->i", AR_LIKE_b, AR_LIKE_b))  # E: Union[numpy.bool_, numpy.ndarray[Any, numpy.dtype[numpy.bool_]]
reveal_type(np.einsum("i,i->i", AR_LIKE_u, AR_LIKE_u))  # E: Union[numpy.unsignedinteger[Any], numpy.ndarray[Any, numpy.dtype[numpy.unsignedinteger[Any]]]
reveal_type(np.einsum("i,i->i", AR_LIKE_i, AR_LIKE_i))  # E: Union[numpy.signedinteger[Any], numpy.ndarray[Any, numpy.dtype[numpy.signedinteger[Any]]]
reveal_type(np.einsum("i,i->i", AR_LIKE_f, AR_LIKE_f))  # E: Union[numpy.floating[Any], numpy.ndarray[Any, numpy.dtype[numpy.floating[Any]]]
reveal_type(np.einsum("i,i->i", AR_LIKE_c, AR_LIKE_c))  # E: Union[numpy.complexfloating[Any, Any], numpy.ndarray[Any, numpy.dtype[numpy.complexfloating[Any, Any]]]
reveal_type(np.einsum("i,i->i", AR_LIKE_b, AR_LIKE_i))  # E: Union[numpy.signedinteger[Any], numpy.ndarray[Any, numpy.dtype[numpy.signedinteger[Any]]]
reveal_type(np.einsum("i,i,i,i->i", AR_LIKE_b, AR_LIKE_u, AR_LIKE_i, AR_LIKE_c))  # E: Union[numpy.complexfloating[Any, Any], numpy.ndarray[Any, numpy.dtype[numpy.complexfloating[Any, Any]]]

reveal_type(np.einsum("i,i->i", AR_LIKE_c, AR_LIKE_c, out=OUT_f))  # E: numpy.ndarray[Any, numpy.dtype[{float64}]
reveal_type(np.einsum("i,i->i", AR_LIKE_U, AR_LIKE_U, dtype=bool, casting="unsafe", out=OUT_f))  # E: numpy.ndarray[Any, numpy.dtype[{float64}]
reveal_type(np.einsum("i,i->i", AR_LIKE_f, AR_LIKE_f, dtype="c16"))  # E: Union[numpy.complexfloating[Any, Any], numpy.ndarray[Any, numpy.dtype[numpy.complexfloating[Any, Any]]]
reveal_type(np.einsum("i,i->i", AR_LIKE_U, AR_LIKE_U, dtype=bool, casting="unsafe"))  # E: Any

reveal_type(np.einsum_path("i,i->i", AR_LIKE_b, AR_LIKE_b))  # E: Tuple[builtins.list[Any], builtins.str]
reveal_type(np.einsum_path("i,i->i", AR_LIKE_u, AR_LIKE_u))  # E: Tuple[builtins.list[Any], builtins.str]
reveal_type(np.einsum_path("i,i->i", AR_LIKE_i, AR_LIKE_i))  # E: Tuple[builtins.list[Any], builtins.str]
reveal_type(np.einsum_path("i,i->i", AR_LIKE_f, AR_LIKE_f))  # E: Tuple[builtins.list[Any], builtins.str]
reveal_type(np.einsum_path("i,i->i", AR_LIKE_c, AR_LIKE_c))  # E: Tuple[builtins.list[Any], builtins.str]
reveal_type(np.einsum_path("i,i->i", AR_LIKE_b, AR_LIKE_i))  # E: Tuple[builtins.list[Any], builtins.str]
reveal_type(np.einsum_path("i,i,i,i->i", AR_LIKE_b, AR_LIKE_u, AR_LIKE_i, AR_LIKE_c))  # E: Tuple[builtins.list[Any], builtins.str]
