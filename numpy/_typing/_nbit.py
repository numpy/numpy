"""A module with the precisions of platform-specific `~numpy.number`s."""

from ._nbit_base import _8Bit, _16Bit, _32Bit, _64Bit, _96Bit, _128Bit

# To-be replaced with a `npt.NBitBase` subclass by numpy's mypy plugin
type _NBitByte = _8Bit
type _NBitShort = _16Bit
type _NBitIntC = _32Bit
type _NBitIntP = _32Bit | _64Bit
type _NBitInt = _NBitIntP
type _NBitLong = _32Bit | _64Bit
type _NBitLongLong = _64Bit

type _NBitHalf = _16Bit
type _NBitSingle = _32Bit
type _NBitDouble = _64Bit
type _NBitLongDouble = _64Bit | _96Bit | _128Bit
