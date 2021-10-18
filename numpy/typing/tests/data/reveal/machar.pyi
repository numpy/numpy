import numpy as np

mach_ar: np.MachAr[np.int64, np.float64]

reveal_type(np.MachAr())  # E: numpy.MachAr[builtins.int, builtins.float]
reveal_type(np.MachAr(float, np.int32, np.float64))  # E: numpy.MachAr[{int32}, {float64}]
reveal_type(np.MachAr(float_to_float=np.float64))  # E: numpy.MachAr[Any, Any]

reveal_type(mach_ar.ibeta)  # E: {int64}
reveal_type(mach_ar.it)  # E: int
reveal_type(mach_ar.negep)  # E: int
reveal_type(mach_ar.epsneg)  # E: {float64}
reveal_type(mach_ar.machep)  # E: int
reveal_type(mach_ar.eps)  # E: {float64}
reveal_type(mach_ar.ngrd)  # E: int
reveal_type(mach_ar.iexp)  # E: int
reveal_type(mach_ar.minexp)  # E: int
reveal_type(mach_ar.xmin)  # E: {float64}
reveal_type(mach_ar.maxexp)  # E: int
reveal_type(mach_ar.xmax)  # E: {float64}
reveal_type(mach_ar.irnd)  # E: int
reveal_type(mach_ar.title)  # E: str
reveal_type(mach_ar.epsilon)  # E: {float64}
reveal_type(mach_ar.tiny)  # E: {float64}
reveal_type(mach_ar.huge)  # E: {float64}
reveal_type(mach_ar.smallest_normal)  # E: {float64}
reveal_type(mach_ar.smallest_subnormal)  # E: {float64}
reveal_type(mach_ar.precision)  # E: int
reveal_type(mach_ar.resolution)  # E: {float64}
