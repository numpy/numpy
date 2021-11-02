import numpy as np
import numpy.typing as npt

AR_f8: npt.NDArray[np.float64]
AR_c16: npt.NDArray[np.complex128]
AR_LIKE_f8: list[float]

reveal_type(np.fft.fftshift(AR_f8))  # E: numpy.ndarray[Any, numpy.dtype[{float64}]]
reveal_type(np.fft.fftshift(AR_LIKE_f8, axes=0))  # E: numpy.ndarray[Any, numpy.dtype[Any]]

reveal_type(np.fft.ifftshift(AR_f8))  # E: numpy.ndarray[Any, numpy.dtype[{float64}]]
reveal_type(np.fft.ifftshift(AR_LIKE_f8, axes=0))  # E: numpy.ndarray[Any, numpy.dtype[Any]]

reveal_type(np.fft.fftfreq(5, AR_f8))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[Any]]]
reveal_type(np.fft.fftfreq(np.int64(), AR_c16))  # E: numpy.ndarray[Any, numpy.dtype[numpy.complexfloating[Any, Any]]]

reveal_type(np.fft.fftfreq(5, AR_f8))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[Any]]]
reveal_type(np.fft.fftfreq(np.int64(), AR_c16))  # E: numpy.ndarray[Any, numpy.dtype[numpy.complexfloating[Any, Any]]]

reveal_type(np.fft.fft(AR_f8))  # E: numpy.ndarray[Any, numpy.dtype[{complex128}]]
reveal_type(np.fft.ifft(AR_f8, axis=1))  # E: numpy.ndarray[Any, numpy.dtype[{complex128}]]
reveal_type(np.fft.rfft(AR_f8, n=None))  # E: numpy.ndarray[Any, numpy.dtype[{complex128}]]
reveal_type(np.fft.irfft(AR_f8, norm="ortho"))  # E: numpy.ndarray[Any, numpy.dtype[{float64}]]
reveal_type(np.fft.hfft(AR_f8, n=2))  # E: numpy.ndarray[Any, numpy.dtype[{float64}]]
reveal_type(np.fft.ihfft(AR_f8))  # E: numpy.ndarray[Any, numpy.dtype[{complex128}]]

reveal_type(np.fft.fftn(AR_f8))  # E: numpy.ndarray[Any, numpy.dtype[{complex128}]]
reveal_type(np.fft.ifftn(AR_f8))  # E: numpy.ndarray[Any, numpy.dtype[{complex128}]]
reveal_type(np.fft.rfftn(AR_f8))  # E: numpy.ndarray[Any, numpy.dtype[{complex128}]]
reveal_type(np.fft.irfftn(AR_f8))  # E: numpy.ndarray[Any, numpy.dtype[{float64}]]

reveal_type(np.fft.rfft2(AR_f8))  # E: numpy.ndarray[Any, numpy.dtype[{complex128}]]
reveal_type(np.fft.ifft2(AR_f8))  # E: numpy.ndarray[Any, numpy.dtype[{complex128}]]
reveal_type(np.fft.fft2(AR_f8))  # E: numpy.ndarray[Any, numpy.dtype[{complex128}]]
reveal_type(np.fft.irfft2(AR_f8))  # E: numpy.ndarray[Any, numpy.dtype[{float64}]]
