# pyright: reportDeprecated=false
# pyright: reportGeneralTypeIssues=false
# mypy: disable-error-code=misc

from typing import final
from typing_extensions import deprecated

# Deprecated in NumPy 2.3, 2025-05-01
@deprecated(
    "`NBitBase` is deprecated and will be removed from numpy.typing in the "
    "future. Use `@typing.overload` or a type parameter with a scalar-type as upper "
    "bound, instead. (deprecated in NumPy 2.3)",
)
@final
class NBitBase: ...

@final
class _256Bit(NBitBase): ...  # type: ignore[deprecated]  # ty: ignore[subclass-of-final-class]

@final
class _128Bit(_256Bit): ...  # ty: ignore[subclass-of-final-class]

@final
class _96Bit(_128Bit): ...  # ty: ignore[subclass-of-final-class]

@final
class _80Bit(_96Bit): ...  # ty: ignore[subclass-of-final-class]

@final
class _64Bit(_80Bit): ...  # ty: ignore[subclass-of-final-class]

@final
class _32Bit(_64Bit): ...  # ty: ignore[subclass-of-final-class]

@final
class _16Bit(_32Bit): ...  # ty: ignore[subclass-of-final-class]

@final
class _8Bit(_16Bit): ...  # ty: ignore[subclass-of-final-class]
