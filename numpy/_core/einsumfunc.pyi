from collections.abc import Sequence
from typing import Any, Literal, overload

import numpy as np
from numpy import _OrderKACF
from numpy._typing import (
    NDArray,
    _ArrayLike,
    _ArrayLikeComplex_co,
    _ArrayLikeObject_co,
    _DTypeLikeComplex_co,
    _DTypeLikeObject,
)

__all__ = ["einsum", "einsum_path"]

type _OptimizeKind = bool | Literal["greedy", "optimal"] | Sequence[Any] | None
type _CastingSafe = Literal["no", "equiv", "safe", "same_kind"]
type _CastingUnsafe = Literal["unsafe"]

# These literals are the 112 most frequent subscript values (per output rank) from an
# AST survey of `einsum` call sites in 167 downstream projects (23_735 call sites,
# 10_171 distinct strings), which cover ~41% of the surveyed calls, plus 11 common
# single- and triple-operand forms.
type _Subscripts0D = Literal[
    "i->",
    "ii",
    ",i->",
    "i,->",
    "i,i",
    "i,i->",
    "ij,ij",
    "ij,ij->",
    "ij,ji->",
    "mo,mo->",
    "nm,nm->",
]
type _Subscripts1D = Literal[
    "ii->i",
    ",i->i",
    "i,->i",
    "i,i->i",
    "ij,j",
    "bn,bn->b",
    "ca,ca->c",
    "ij,ij->i",
    "ij,ij->j",
    "ij,ji->i",
    "ij,ki->k",
    "ik,ik->k",
    "ji,ij->j",
    "ni,ni->n",
    "ny,ny->y",
    "yx,xy->y",
]
type _Subscripts2D = Literal[
    "ij->ji",
    "ji",
    "aabb->ab",
    "abbc",
    "abcb",
    "ijik->jk",
    "ijkl->ij",
    "ijkl->jl",
    "illj",
    "llij",
    "prrs",
    "b,a->ba",
    "i,j",
    "i,j->ij",
    "i,j->ji",
    "m,d->md",
    "ab,ab->ab",
    "bi,bj->ij",
    "bi,fb->if",
    "ij,ik->jk",
    "ij,jk",
    "ij,jk->ik",
    "ij,ki->ki",
    "ij,kj->ik",
    "ik,jk->ij",
    "ik,kj->ij",
    "ji,jk->ik",
    "ji,ki->jk",
    "ki,kj->ij",
    "nk,km->nm",
    "nm,no->mo",
    "nw,nx->wx",
    "ij,ijk->ij",
    "ij,ijk->ik",
    "ij,ijk->jk",
    "ij,ikj->ik",
    "ik,ijk->ij",
    "jk,ijk->ij",
    "ijk,ij->ik",
    "ijk,ik->ij",
    "ijk,jk->ik",
    "ijk,kj->ki",
    "ikj,ik->ij",
    "mnq,nm->mq",
    "nij,nj->ni",
    "nmq,nm->nq",
    "nmq,nq->nm",
    "svt,vt->st",
    "ajk,ajl->kl",
    "ijk,ijk->ij",
    "ijk,ijk->ik",
    "ijk,jil->kl",
    "ijk,jlk->il",
    "ijk,lkj->il",
    "ijl,jik->kl",
    "kij,kil",
    "nft,ftm->nm",
]
type _Subscripts3D = Literal[
    "ab,bso->aso",
    "cl,cpx->lpx",
    "cv,zac->zav",
    "gi,ghj->hij",
    "ij,jab->iab",
    "ij,njk->nik",
    "nm,ftm->nft",
    "qp,iaq->iap",
    "ijk,jk->ijk",
    "ijk,jl",
    "ijk,mk->ijm",
    "nft,nm->ftm",
    "nij,jk->nik",
    "rtp,pm->rtm",
    "zac,cv->zav",
    "aij,ajk->aik",
    "aij,jka->aik",
    "bij,bjk->bik",
    "btk,bnk->btn",
    "fij,ftj->fti",
    "ija,ajk->aik",
    "ijk,ikl->ijl",
    "ijk,ikm->ijm",
    "ijk,jkl->ikl",
    "nfk,nft->nkt",
    "nfm,ftm->nft",
    "nft,ftm->nfm",
    "nft,nfm->ftm",
    "nij,njk->nik",
    "nkt,nft->nfk",
    "tij,tjk->tik",
    "i,j,k->ijk",
]
type _Subscripts4D = Literal[
    "abcd,cdjk->abjk",
    "badc,cdjk->abjk",
    "BNTS,BSNH->BTNH",
    "BTNH,BSNH->BNTS",
    "iAjB,AkBl->ikjl",
    "ijmn,ijnk->ijmk",
    "tpqi,trsi->pqrs",
]

###

@overload  # 0d T, optimize=False
def einsum[ScalarT: np.bool | np.number](
    subscripts: _Subscripts0D,
    /,
    *operands: _ArrayLike[ScalarT],
    out: None = None,
    dtype: None = None,
    order: _OrderKACF = "K",
    casting: _CastingSafe = "safe",
    optimize: Literal[False] = False,
) -> ScalarT: ...
@overload  # 0d T, optimize=<given>
def einsum[ScalarT: np.bool | np.number](
    subscripts: _Subscripts0D,
    /,
    *operands: _ArrayLike[ScalarT],
    out: None = None,
    dtype: None = None,
    order: _OrderKACF = "K",
    casting: _CastingSafe = "safe",
    optimize: Literal[True, "greedy", "optimal"] | Sequence[Any] | None,
) -> ScalarT | np.ndarray[tuple[()], np.dtype[ScalarT]]: ...
@overload  # 1d T
def einsum[ScalarT: np.bool | np.number | np.object_](
    subscripts: _Subscripts1D,
    /,
    *operands: _ArrayLike[ScalarT],
    out: None = None,
    dtype: None = None,
    order: _OrderKACF = "K",
    casting: _CastingSafe = "safe",
    optimize: _OptimizeKind = False,
) -> np.ndarray[tuple[int], np.dtype[ScalarT]]: ...
@overload  # 2d T
def einsum[ScalarT: np.bool | np.number | np.object_](
    subscripts: _Subscripts2D,
    /,
    *operands: _ArrayLike[ScalarT],
    out: None = None,
    dtype: None = None,
    order: _OrderKACF = "K",
    casting: _CastingSafe = "safe",
    optimize: _OptimizeKind = False,
) -> np.ndarray[tuple[int, int], np.dtype[ScalarT]]: ...
@overload  # 3d T
def einsum[ScalarT: np.bool | np.number | np.object_](
    subscripts: _Subscripts3D,
    /,
    *operands: _ArrayLike[ScalarT],
    out: None = None,
    dtype: None = None,
    order: _OrderKACF = "K",
    casting: _CastingSafe = "safe",
    optimize: _OptimizeKind = False,
) -> np.ndarray[tuple[int, int, int], np.dtype[ScalarT]]: ...
@overload  # 4d T
def einsum[ScalarT: np.bool | np.number | np.object_](
    subscripts: _Subscripts4D,
    /,
    *operands: _ArrayLike[ScalarT],
    out: None = None,
    dtype: None = None,
    order: _OrderKACF = "K",
    casting: _CastingSafe = "safe",
    optimize: _OptimizeKind = False,
) -> np.ndarray[tuple[int, int, int, int], np.dtype[ScalarT]]: ...
@overload  # ?d
def einsum(
    subscripts: str | _ArrayLikeComplex_co,
    /,
    *operands: _ArrayLikeComplex_co | _ArrayLikeObject_co,
    out: None = None,
    dtype: _DTypeLikeComplex_co | _DTypeLikeObject | None = None,
    order: _OrderKACF = "K",
    casting: _CastingSafe = "safe",
    optimize: _OptimizeKind = False,
) -> Any: ...
@overload  # ?d, casting="unsafe"
def einsum(
    subscripts: str | _ArrayLikeComplex_co,
    /,
    *operands: Any,
    out: None = None,
    dtype: _DTypeLikeComplex_co | _DTypeLikeObject | None = None,
    order: _OrderKACF = "K",
    casting: _CastingUnsafe,
    optimize: _OptimizeKind = False,
) -> Any: ...
@overload  # ?d, out=<given>
def einsum[OutT: NDArray[np.bool | np.number | np.object_]](
    subscripts: str | _ArrayLikeComplex_co,
    /,
    *operands: _ArrayLikeComplex_co | _ArrayLikeObject_co,
    out: OutT,
    dtype: _DTypeLikeComplex_co | _DTypeLikeObject | None = None,
    order: _OrderKACF = "K",
    casting: _CastingSafe = "safe",
    optimize: _OptimizeKind = False,
) -> OutT: ...
@overload  # ?d, out=<given>, casting="unsafe"
def einsum[OutT: NDArray[np.bool | np.number | np.object_]](
    subscripts: str | _ArrayLikeComplex_co,
    /,
    *operands: Any,
    out: OutT,
    dtype: _DTypeLikeComplex_co | _DTypeLikeObject | None = None,
    order: _OrderKACF = "K",
    casting: _CastingUnsafe,
    optimize: _OptimizeKind = False,
) -> OutT: ...

# NOTE: In practice the list consists of a `str` (first element)
# and a variable number of integer tuples.
def einsum_path(
    subscripts: str | _ArrayLikeComplex_co,
    /,
    *operands: _ArrayLikeComplex_co | _ArrayLikeObject_co,
    optimize: _OptimizeKind = "greedy",
    einsum_call: Literal[False] = False,
) -> tuple[list[Any], str]: ...
