import numpy as np
import numpy.polynomial as npp
import numpy.typing as npt

AR_f8: npt.NDArray[np.float64]
AR_c16: npt.NDArray[np.complex128]
AR_O: npt.NDArray[np.object_]
AR_U: npt.NDArray[np.str_]

poly_obj: npp.polynomial.Polynomial

npp.polynomial.polymul(AR_f8, AR_U)  # E: incompatible type
npp.polynomial.polydiv(AR_f8, AR_U)  # E: incompatible type

5**poly_obj  # E: No overload variant

npp.polynomial.polyint(AR_U)  # E: incompatible type
npp.polynomial.polyint(AR_f8, m=1j)  # E: No overload variant

npp.polynomial.polyder(AR_U)  # E: incompatible type
npp.polynomial.polyder(AR_f8, m=1j)  # E: No overload variant

npp.polynomial.polyfit(AR_O, AR_f8, 1)  # E: incompatible type
npp.polynomial.polyfit(AR_f8, AR_f8, 1, rcond=1j)  # E: No overload variant
npp.polynomial.polyfit(AR_f8, AR_f8, 1, w=AR_c16)  # E: incompatible type
npp.polynomial.polyfit(AR_f8, AR_f8, 1, cov="bob")  # E: No overload variant

npp.polynomial.polyval(AR_f8, AR_U)  # E: incompatible type
npp.polynomial.polyadd(AR_f8, AR_U)  # E: incompatible type
npp.polynomial.polysub(AR_f8, AR_U)  # E: incompatible type
