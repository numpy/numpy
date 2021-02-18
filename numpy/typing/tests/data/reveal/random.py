from __future__ import annotations

from typing import Any, List

import numpy as np

def_rng = np.random.default_rng()
seed_seq = np.random.SeedSequence()
mt19937 = np.random.MT19937()
pcg64 = np.random.PCG64()
sfc64 = np.random.SFC64()
philox = np.random.Philox()
seedless_seq = np.random.bit_generator.SeedlessSeedSequence()

reveal_type(def_rng)  # E: numpy.random._generator.Generator
reveal_type(mt19937)  # E: numpy.random._mt19937.MT19937
reveal_type(pcg64)  # E: numpy.random._pcg64.PCG64
reveal_type(sfc64)  # E: numpy.random._sfc64.SFC64
reveal_type(philox)  # E: numpy.random._philox.Philox
reveal_type(seed_seq)  # E: numpy.random.bit_generator.SeedSequence
reveal_type(seedless_seq)  # E: numpy.random.bit_generator.SeedlessSeedSequence

mt19937_jumped = mt19937.jumped()
mt19937_jumped3 = mt19937.jumped(3)
mt19937_raw = mt19937.random_raw()
mt19937_raw_arr = mt19937.random_raw(5)

reveal_type(mt19937_jumped)  # E: numpy.random._mt19937.MT19937
reveal_type(mt19937_jumped3)  # E: numpy.random._mt19937.MT19937
reveal_type(mt19937_raw)  # E: int
reveal_type(mt19937_raw_arr)  # E: numpy.ndarray[Any, numpy.dtype[numpy.unsignedinteger[numpy.typing._64Bit]]]
reveal_type(mt19937.lock)  # E: threading.Lock

pcg64_jumped = pcg64.jumped()
pcg64_jumped3 = pcg64.jumped(3)
pcg64_adv = pcg64.advance(3)
pcg64_raw = pcg64.random_raw()
pcg64_raw_arr = pcg64.random_raw(5)

reveal_type(pcg64_jumped)  # E: numpy.random._pcg64.PCG64
reveal_type(pcg64_jumped3)  # E: numpy.random._pcg64.PCG64
reveal_type(pcg64_adv)  # E: numpy.random._pcg64.PCG64
reveal_type(pcg64_raw)  # E: int
reveal_type(pcg64_raw_arr)  # E: numpy.ndarray[Any, numpy.dtype[numpy.unsignedinteger[numpy.typing._64Bit]]]
reveal_type(pcg64.lock)  # E: threading.Lock

philox_jumped = philox.jumped()
philox_jumped3 = philox.jumped(3)
philox_adv = philox.advance(3)
philox_raw = philox.random_raw()
philox_raw_arr = philox.random_raw(5)

reveal_type(philox_jumped)  # E: numpy.random._philox.Philox
reveal_type(philox_jumped3)  # E: numpy.random._philox.Philox
reveal_type(philox_adv)  # E: numpy.random._philox.Philox
reveal_type(philox_raw)  # E: int
reveal_type(philox_raw_arr)  # E: numpy.ndarray[Any, numpy.dtype[numpy.unsignedinteger[numpy.typing._64Bit]]]
reveal_type(philox.lock)  # E: threading.Lock

sfc64_raw = sfc64.random_raw()
sfc64_raw_arr = sfc64.random_raw(5)

reveal_type(sfc64_raw)  # E: int
reveal_type(sfc64_raw_arr)  # E: numpy.ndarray[Any, numpy.dtype[numpy.unsignedinteger[numpy.typing._64Bit]]]
reveal_type(sfc64.lock)  # E: threading.Lock

reveal_type(seed_seq.pool)  # numpy.ndarray[Any, numpy.dtype[numpy.unsignedinteger[numpy.typing._32Bit]]]
reveal_type(seed_seq.entropy)  # E:Union[None, int, Sequence[int]]
reveal_type(seed_seq.spawn(1))  # E: list[numpy.random.bit_generator.SeedSequence]
reveal_type(seed_seq.generate_state(8, "uint32"))  # E: numpy.ndarray[Any, numpy.dtype[Union[numpy.unsignedinteger[numpy.typing._32Bit], numpy.unsignedinteger[numpy.typing._64Bit]]]]
reveal_type(seed_seq.generate_state(8, "uint64"))  # E: numpy.ndarray[Any, numpy.dtype[Union[numpy.unsignedinteger[numpy.typing._32Bit], numpy.unsignedinteger[numpy.typing._64Bit]]]]


def_gen: np.random.Generator = np.random.default_rng()

D_arr_0p1: np.ndarray[Any, np.dtype[np.float64]] = np.array([0.1])
D_arr_0p5: np.ndarray[Any, np.dtype[np.float64]] = np.array([0.5])
D_arr_0p9: np.ndarray[Any, np.dtype[np.float64]] = np.array([0.9])
D_arr_1p5: np.ndarray[Any, np.dtype[np.float64]] = np.array([1.5])
I_arr_10: np.ndarray[Any, np.dtype[np.int_]] = np.array([10], dtype=np.int_)
I_arr_20: np.ndarray[Any, np.dtype[np.int_]] = np.array([20], dtype=np.int_)
D_arr_like_0p1: List[float] = [0.1]
D_arr_like_0p5: List[float] = [0.5]
D_arr_like_0p9: List[float] = [0.9]
D_arr_like_1p5: List[float] = [1.5]
I_arr_like_10: List[int] = [10]
I_arr_like_20: List[int] = [20]
D_2D_like: List[List[float]] = [[1, 2], [2, 3], [3, 4], [4, 5.1]]
D_2D: np.ndarray[Any, np.dtype[np.float64]] = np.array(D_2D_like)

reveal_type(def_gen.standard_normal())  # E: float
reveal_type(def_gen.standard_normal(size=None))  # E: float
reveal_type(def_gen.standard_normal(size=1))  # E: numpy.ndarray[Any, numpy.dtype[Union[numpy.floating[numpy.typing._32Bit], numpy.floating[numpy.typing._64Bit]]]

reveal_type(def_gen.random())  # E: float
reveal_type(def_gen.random(size=None))  # E: float
reveal_type(def_gen.random(size=1))  # E: numpy.ndarray[Any, numpy.dtype[Union[numpy.floating[numpy.typing._32Bit], numpy.floating[numpy.typing._64Bit]]]

reveal_type(def_gen.standard_cauchy())  # E: float
reveal_type(def_gen.standard_cauchy(size=None))  # E: float
reveal_type(def_gen.standard_cauchy(size=1))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]

reveal_type(def_gen.standard_exponential())  # E: float
reveal_type(def_gen.standard_exponential(size=None))  # E: float
reveal_type(def_gen.standard_exponential(size=1))  # E: numpy.ndarray[Any, numpy.dtype[Union[numpy.floating[numpy.typing._32Bit], numpy.floating[numpy.typing._64Bit]]]

reveal_type(def_gen.zipf(1.5))  # E: int
reveal_type(def_gen.zipf(1.5, size=None))  # E: int
reveal_type(def_gen.zipf(1.5, size=1))  # E: numpy.ndarray[Any, numpy.dtype[numpy.signedinteger[numpy.typing._64Bit]]
reveal_type(def_gen.zipf(D_arr_1p5))  # E: numpy.ndarray[Any, numpy.dtype[numpy.signedinteger[numpy.typing._64Bit]]
reveal_type(def_gen.zipf(D_arr_1p5, size=1))  # E: numpy.ndarray[Any, numpy.dtype[numpy.signedinteger[numpy.typing._64Bit]]
reveal_type(def_gen.zipf(D_arr_like_1p5))  # E: numpy.ndarray[Any, numpy.dtype[numpy.signedinteger[numpy.typing._64Bit]]
reveal_type(def_gen.zipf(D_arr_like_1p5, size=1))  # E: numpy.ndarray[Any, numpy.dtype[numpy.signedinteger[numpy.typing._64Bit]]

reveal_type(def_gen.weibull(0.5))  # E: float
reveal_type(def_gen.weibull(0.5, size=None))  # E: float
reveal_type(def_gen.weibull(0.5, size=1))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.weibull(D_arr_0p5))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.weibull(D_arr_0p5, size=1))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.weibull(D_arr_like_0p5))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.weibull(D_arr_like_0p5, size=1))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]

reveal_type(def_gen.standard_t(0.5))  # E: float
reveal_type(def_gen.standard_t(0.5, size=None))  # E: float
reveal_type(def_gen.standard_t(0.5, size=1))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.standard_t(D_arr_0p5))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.standard_t(D_arr_0p5, size=1))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.standard_t(D_arr_like_0p5))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.standard_t(D_arr_like_0p5, size=1))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]

reveal_type(def_gen.poisson(0.5))  # E: int
reveal_type(def_gen.poisson(0.5, size=None))  # E: int
reveal_type(def_gen.poisson(0.5, size=1))  # E: numpy.ndarray[Any, numpy.dtype[numpy.signedinteger[numpy.typing._64Bit]]
reveal_type(def_gen.poisson(D_arr_0p5))  # E: numpy.ndarray[Any, numpy.dtype[numpy.signedinteger[numpy.typing._64Bit]]
reveal_type(def_gen.poisson(D_arr_0p5, size=1))  # E: numpy.ndarray[Any, numpy.dtype[numpy.signedinteger[numpy.typing._64Bit]]
reveal_type(def_gen.poisson(D_arr_like_0p5))  # E: numpy.ndarray[Any, numpy.dtype[numpy.signedinteger[numpy.typing._64Bit]]
reveal_type(def_gen.poisson(D_arr_like_0p5, size=1))  # E: numpy.ndarray[Any, numpy.dtype[numpy.signedinteger[numpy.typing._64Bit]]

reveal_type(def_gen.power(0.5))  # E: float
reveal_type(def_gen.power(0.5, size=None))  # E: float
reveal_type(def_gen.power(0.5, size=1))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.power(D_arr_0p5))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.power(D_arr_0p5, size=1))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.power(D_arr_like_0p5))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.power(D_arr_like_0p5, size=1))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]

reveal_type(def_gen.pareto(0.5))  # E: float
reveal_type(def_gen.pareto(0.5, size=None))  # E: float
reveal_type(def_gen.pareto(0.5, size=1))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.pareto(D_arr_0p5))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.pareto(D_arr_0p5, size=1))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.pareto(D_arr_like_0p5))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.pareto(D_arr_like_0p5, size=1))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]

reveal_type(def_gen.chisquare(0.5))  # E: float
reveal_type(def_gen.chisquare(0.5, size=None))  # E: float
reveal_type(def_gen.chisquare(0.5, size=1))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.chisquare(D_arr_0p5))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.chisquare(D_arr_0p5, size=1))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.chisquare(D_arr_like_0p5))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.chisquare(D_arr_like_0p5, size=1))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]

reveal_type(def_gen.exponential(0.5))  # E: float
reveal_type(def_gen.exponential(0.5, size=None))  # E: float
reveal_type(def_gen.exponential(0.5, size=1))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.exponential(D_arr_0p5))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.exponential(D_arr_0p5, size=1))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.exponential(D_arr_like_0p5))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.exponential(D_arr_like_0p5, size=1))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]

reveal_type(def_gen.geometric(0.5))  # E: int
reveal_type(def_gen.geometric(0.5, size=None))  # E: int
reveal_type(def_gen.geometric(0.5, size=1))  # E: numpy.ndarray[Any, numpy.dtype[numpy.signedinteger[numpy.typing._64Bit]]
reveal_type(def_gen.geometric(D_arr_0p5))  # E: numpy.ndarray[Any, numpy.dtype[numpy.signedinteger[numpy.typing._64Bit]]
reveal_type(def_gen.geometric(D_arr_0p5, size=1))  # E: numpy.ndarray[Any, numpy.dtype[numpy.signedinteger[numpy.typing._64Bit]]
reveal_type(def_gen.geometric(D_arr_like_0p5))  # E: numpy.ndarray[Any, numpy.dtype[numpy.signedinteger[numpy.typing._64Bit]]
reveal_type(def_gen.geometric(D_arr_like_0p5, size=1))  # E: numpy.ndarray[Any, numpy.dtype[numpy.signedinteger[numpy.typing._64Bit]]

reveal_type(def_gen.logseries(0.5))  # E: int
reveal_type(def_gen.logseries(0.5, size=None))  # E: int
reveal_type(def_gen.logseries(0.5, size=1))  # E: numpy.ndarray[Any, numpy.dtype[numpy.signedinteger[numpy.typing._64Bit]]
reveal_type(def_gen.logseries(D_arr_0p5))  # E: numpy.ndarray[Any, numpy.dtype[numpy.signedinteger[numpy.typing._64Bit]]
reveal_type(def_gen.logseries(D_arr_0p5, size=1))  # E: numpy.ndarray[Any, numpy.dtype[numpy.signedinteger[numpy.typing._64Bit]]
reveal_type(def_gen.logseries(D_arr_like_0p5))  # E: numpy.ndarray[Any, numpy.dtype[numpy.signedinteger[numpy.typing._64Bit]]
reveal_type(def_gen.logseries(D_arr_like_0p5, size=1))  # E: numpy.ndarray[Any, numpy.dtype[numpy.signedinteger[numpy.typing._64Bit]]

reveal_type(def_gen.rayleigh(0.5))  # E: float
reveal_type(def_gen.rayleigh(0.5, size=None))  # E: float
reveal_type(def_gen.rayleigh(0.5, size=1))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.rayleigh(D_arr_0p5))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.rayleigh(D_arr_0p5, size=1))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.rayleigh(D_arr_like_0p5))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.rayleigh(D_arr_like_0p5, size=1))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]

reveal_type(def_gen.standard_gamma(0.5))  # E: float
reveal_type(def_gen.standard_gamma(0.5, size=None))  # E: float
reveal_type(def_gen.standard_gamma(0.5, size=1))  # E: numpy.ndarray[Any, numpy.dtype[Union[numpy.floating[numpy.typing._32Bit], numpy.floating[numpy.typing._64Bit]]]
reveal_type(def_gen.standard_gamma(D_arr_0p5))  # E: numpy.ndarray[Any, numpy.dtype[Union[numpy.floating[numpy.typing._32Bit], numpy.floating[numpy.typing._64Bit]]]
reveal_type(def_gen.standard_gamma(D_arr_0p5, size=1))  # E: numpy.ndarray[Any, numpy.dtype[Union[numpy.floating[numpy.typing._32Bit], numpy.floating[numpy.typing._64Bit]]]
reveal_type(def_gen.standard_gamma(D_arr_like_0p5))  # E: numpy.ndarray[Any, numpy.dtype[Union[numpy.floating[numpy.typing._32Bit], numpy.floating[numpy.typing._64Bit]]]
reveal_type(def_gen.standard_gamma(D_arr_like_0p5, size=1))  # E: numpy.ndarray[Any, numpy.dtype[Union[numpy.floating[numpy.typing._32Bit], numpy.floating[numpy.typing._64Bit]]]

reveal_type(def_gen.vonmises(0.5, 0.5))  # E: float
reveal_type(def_gen.vonmises(0.5, 0.5, size=None))  # E: float
reveal_type(def_gen.vonmises(0.5, 0.5, size=1))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.vonmises(D_arr_0p5, 0.5))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.vonmises(0.5, D_arr_0p5))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.vonmises(D_arr_0p5, 0.5, size=1))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.vonmises(0.5, D_arr_0p5, size=1))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.vonmises(D_arr_like_0p5, 0.5))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.vonmises(0.5, D_arr_like_0p5))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.vonmises(D_arr_0p5, D_arr_0p5))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.vonmises(D_arr_like_0p5, D_arr_like_0p5))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.vonmises(D_arr_0p5, D_arr_0p5, size=1))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.vonmises(D_arr_like_0p5, D_arr_like_0p5, size=1))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]

reveal_type(def_gen.wald(0.5, 0.5))  # E: float
reveal_type(def_gen.wald(0.5, 0.5, size=None))  # E: float
reveal_type(def_gen.wald(0.5, 0.5, size=1))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.wald(D_arr_0p5, 0.5))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.wald(0.5, D_arr_0p5))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.wald(D_arr_0p5, 0.5, size=1))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.wald(0.5, D_arr_0p5, size=1))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.wald(D_arr_like_0p5, 0.5))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.wald(0.5, D_arr_like_0p5))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.wald(D_arr_0p5, D_arr_0p5))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.wald(D_arr_like_0p5, D_arr_like_0p5))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.wald(D_arr_0p5, D_arr_0p5, size=1))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.wald(D_arr_like_0p5, D_arr_like_0p5, size=1))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]

reveal_type(def_gen.uniform(0.5, 0.5))  # E: float
reveal_type(def_gen.uniform(0.5, 0.5, size=None))  # E: float
reveal_type(def_gen.uniform(0.5, 0.5, size=1))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.uniform(D_arr_0p5, 0.5))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.uniform(0.5, D_arr_0p5))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.uniform(D_arr_0p5, 0.5, size=1))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.uniform(0.5, D_arr_0p5, size=1))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.uniform(D_arr_like_0p5, 0.5))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.uniform(0.5, D_arr_like_0p5))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.uniform(D_arr_0p5, D_arr_0p5))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.uniform(D_arr_like_0p5, D_arr_like_0p5))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.uniform(D_arr_0p5, D_arr_0p5, size=1))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.uniform(D_arr_like_0p5, D_arr_like_0p5, size=1))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]

reveal_type(def_gen.beta(0.5, 0.5))  # E: float
reveal_type(def_gen.beta(0.5, 0.5, size=None))  # E: float
reveal_type(def_gen.beta(0.5, 0.5, size=1))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.beta(D_arr_0p5, 0.5))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.beta(0.5, D_arr_0p5))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.beta(D_arr_0p5, 0.5, size=1))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.beta(0.5, D_arr_0p5, size=1))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.beta(D_arr_like_0p5, 0.5))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.beta(0.5, D_arr_like_0p5))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.beta(D_arr_0p5, D_arr_0p5))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.beta(D_arr_like_0p5, D_arr_like_0p5))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.beta(D_arr_0p5, D_arr_0p5, size=1))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.beta(D_arr_like_0p5, D_arr_like_0p5, size=1))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]

reveal_type(def_gen.f(0.5, 0.5))  # E: float
reveal_type(def_gen.f(0.5, 0.5, size=None))  # E: float
reveal_type(def_gen.f(0.5, 0.5, size=1))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.f(D_arr_0p5, 0.5))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.f(0.5, D_arr_0p5))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.f(D_arr_0p5, 0.5, size=1))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.f(0.5, D_arr_0p5, size=1))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.f(D_arr_like_0p5, 0.5))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.f(0.5, D_arr_like_0p5))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.f(D_arr_0p5, D_arr_0p5))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.f(D_arr_like_0p5, D_arr_like_0p5))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.f(D_arr_0p5, D_arr_0p5, size=1))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.f(D_arr_like_0p5, D_arr_like_0p5, size=1))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]

reveal_type(def_gen.gamma(0.5, 0.5))  # E: float
reveal_type(def_gen.gamma(0.5, 0.5, size=None))  # E: float
reveal_type(def_gen.gamma(0.5, 0.5, size=1))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.gamma(D_arr_0p5, 0.5))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.gamma(0.5, D_arr_0p5))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.gamma(D_arr_0p5, 0.5, size=1))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.gamma(0.5, D_arr_0p5, size=1))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.gamma(D_arr_like_0p5, 0.5))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.gamma(0.5, D_arr_like_0p5))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.gamma(D_arr_0p5, D_arr_0p5))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.gamma(D_arr_like_0p5, D_arr_like_0p5))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.gamma(D_arr_0p5, D_arr_0p5, size=1))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.gamma(D_arr_like_0p5, D_arr_like_0p5, size=1))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]

reveal_type(def_gen.gumbel(0.5, 0.5))  # E: float
reveal_type(def_gen.gumbel(0.5, 0.5, size=None))  # E: float
reveal_type(def_gen.gumbel(0.5, 0.5, size=1))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.gumbel(D_arr_0p5, 0.5))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.gumbel(0.5, D_arr_0p5))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.gumbel(D_arr_0p5, 0.5, size=1))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.gumbel(0.5, D_arr_0p5, size=1))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.gumbel(D_arr_like_0p5, 0.5))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.gumbel(0.5, D_arr_like_0p5))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.gumbel(D_arr_0p5, D_arr_0p5))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.gumbel(D_arr_like_0p5, D_arr_like_0p5))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.gumbel(D_arr_0p5, D_arr_0p5, size=1))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.gumbel(D_arr_like_0p5, D_arr_like_0p5, size=1))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]

reveal_type(def_gen.laplace(0.5, 0.5))  # E: float
reveal_type(def_gen.laplace(0.5, 0.5, size=None))  # E: float
reveal_type(def_gen.laplace(0.5, 0.5, size=1))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.laplace(D_arr_0p5, 0.5))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.laplace(0.5, D_arr_0p5))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.laplace(D_arr_0p5, 0.5, size=1))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.laplace(0.5, D_arr_0p5, size=1))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.laplace(D_arr_like_0p5, 0.5))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.laplace(0.5, D_arr_like_0p5))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.laplace(D_arr_0p5, D_arr_0p5))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.laplace(D_arr_like_0p5, D_arr_like_0p5))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.laplace(D_arr_0p5, D_arr_0p5, size=1))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.laplace(D_arr_like_0p5, D_arr_like_0p5, size=1))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]

reveal_type(def_gen.logistic(0.5, 0.5))  # E: float
reveal_type(def_gen.logistic(0.5, 0.5, size=None))  # E: float
reveal_type(def_gen.logistic(0.5, 0.5, size=1))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.logistic(D_arr_0p5, 0.5))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.logistic(0.5, D_arr_0p5))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.logistic(D_arr_0p5, 0.5, size=1))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.logistic(0.5, D_arr_0p5, size=1))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.logistic(D_arr_like_0p5, 0.5))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.logistic(0.5, D_arr_like_0p5))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.logistic(D_arr_0p5, D_arr_0p5))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.logistic(D_arr_like_0p5, D_arr_like_0p5))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.logistic(D_arr_0p5, D_arr_0p5, size=1))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.logistic(D_arr_like_0p5, D_arr_like_0p5, size=1))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]

reveal_type(def_gen.lognormal(0.5, 0.5))  # E: float
reveal_type(def_gen.lognormal(0.5, 0.5, size=None))  # E: float
reveal_type(def_gen.lognormal(0.5, 0.5, size=1))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.lognormal(D_arr_0p5, 0.5))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.lognormal(0.5, D_arr_0p5))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.lognormal(D_arr_0p5, 0.5, size=1))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.lognormal(0.5, D_arr_0p5, size=1))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.lognormal(D_arr_like_0p5, 0.5))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.lognormal(0.5, D_arr_like_0p5))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.lognormal(D_arr_0p5, D_arr_0p5))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.lognormal(D_arr_like_0p5, D_arr_like_0p5))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.lognormal(D_arr_0p5, D_arr_0p5, size=1))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.lognormal(D_arr_like_0p5, D_arr_like_0p5, size=1))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]

reveal_type(def_gen.noncentral_chisquare(0.5, 0.5))  # E: float
reveal_type(def_gen.noncentral_chisquare(0.5, 0.5, size=None))  # E: float
reveal_type(def_gen.noncentral_chisquare(0.5, 0.5, size=1))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.noncentral_chisquare(D_arr_0p5, 0.5))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.noncentral_chisquare(0.5, D_arr_0p5))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.noncentral_chisquare(D_arr_0p5, 0.5, size=1))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.noncentral_chisquare(0.5, D_arr_0p5, size=1))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.noncentral_chisquare(D_arr_like_0p5, 0.5))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.noncentral_chisquare(0.5, D_arr_like_0p5))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.noncentral_chisquare(D_arr_0p5, D_arr_0p5))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.noncentral_chisquare(D_arr_like_0p5, D_arr_like_0p5))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.noncentral_chisquare(D_arr_0p5, D_arr_0p5, size=1))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.noncentral_chisquare(D_arr_like_0p5, D_arr_like_0p5, size=1))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]

reveal_type(def_gen.normal(0.5, 0.5))  # E: float
reveal_type(def_gen.normal(0.5, 0.5, size=None))  # E: float
reveal_type(def_gen.normal(0.5, 0.5, size=1))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.normal(D_arr_0p5, 0.5))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.normal(0.5, D_arr_0p5))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.normal(D_arr_0p5, 0.5, size=1))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.normal(0.5, D_arr_0p5, size=1))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.normal(D_arr_like_0p5, 0.5))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.normal(0.5, D_arr_like_0p5))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.normal(D_arr_0p5, D_arr_0p5))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.normal(D_arr_like_0p5, D_arr_like_0p5))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.normal(D_arr_0p5, D_arr_0p5, size=1))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.normal(D_arr_like_0p5, D_arr_like_0p5, size=1))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]

reveal_type(def_gen.triangular(0.1, 0.5, 0.9))  # E: float
reveal_type(def_gen.triangular(0.1, 0.5, 0.9, size=None))  # E: float
reveal_type(def_gen.triangular(0.1, 0.5, 0.9, size=1))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.triangular(D_arr_0p1, 0.5, 0.9))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.triangular(0.1, D_arr_0p5, 0.9))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.triangular(D_arr_0p1, 0.5, D_arr_like_0p9, size=1))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.triangular(0.1, D_arr_0p5, 0.9, size=1))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.triangular(D_arr_like_0p1, 0.5, D_arr_0p9))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.triangular(0.5, D_arr_like_0p5, 0.9))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.triangular(D_arr_0p1, D_arr_0p5, 0.9))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.triangular(D_arr_like_0p1, D_arr_like_0p5, 0.9))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.triangular(D_arr_0p1, D_arr_0p5, D_arr_0p9, size=1))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.triangular(D_arr_like_0p1, D_arr_like_0p5, D_arr_like_0p9, size=1))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]

reveal_type(def_gen.noncentral_f(0.1, 0.5, 0.9))  # E: float
reveal_type(def_gen.noncentral_f(0.1, 0.5, 0.9, size=None))  # E: float
reveal_type(def_gen.noncentral_f(0.1, 0.5, 0.9, size=1))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.noncentral_f(D_arr_0p1, 0.5, 0.9))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.noncentral_f(0.1, D_arr_0p5, 0.9))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.noncentral_f(D_arr_0p1, 0.5, D_arr_like_0p9, size=1))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.noncentral_f(0.1, D_arr_0p5, 0.9, size=1))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.noncentral_f(D_arr_like_0p1, 0.5, D_arr_0p9))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.noncentral_f(0.5, D_arr_like_0p5, 0.9))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.noncentral_f(D_arr_0p1, D_arr_0p5, 0.9))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.noncentral_f(D_arr_like_0p1, D_arr_like_0p5, 0.9))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.noncentral_f(D_arr_0p1, D_arr_0p5, D_arr_0p9, size=1))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.noncentral_f(D_arr_like_0p1, D_arr_like_0p5, D_arr_like_0p9, size=1))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]

reveal_type(def_gen.binomial(10, 0.5))  # E: int
reveal_type(def_gen.binomial(10, 0.5, size=None))  # E: int
reveal_type(def_gen.binomial(10, 0.5, size=1))  # E: numpy.ndarray[Any, numpy.dtype[numpy.signedinteger[numpy.typing._64Bit]]
reveal_type(def_gen.binomial(I_arr_10, 0.5))  # E: numpy.ndarray[Any, numpy.dtype[numpy.signedinteger[numpy.typing._64Bit]]
reveal_type(def_gen.binomial(10, D_arr_0p5))  # E: numpy.ndarray[Any, numpy.dtype[numpy.signedinteger[numpy.typing._64Bit]]
reveal_type(def_gen.binomial(I_arr_10, 0.5, size=1))  # E: numpy.ndarray[Any, numpy.dtype[numpy.signedinteger[numpy.typing._64Bit]]
reveal_type(def_gen.binomial(10, D_arr_0p5, size=1))  # E: numpy.ndarray[Any, numpy.dtype[numpy.signedinteger[numpy.typing._64Bit]]
reveal_type(def_gen.binomial(I_arr_like_10, 0.5))  # E: numpy.ndarray[Any, numpy.dtype[numpy.signedinteger[numpy.typing._64Bit]]
reveal_type(def_gen.binomial(10, D_arr_like_0p5))  # E: numpy.ndarray[Any, numpy.dtype[numpy.signedinteger[numpy.typing._64Bit]]
reveal_type(def_gen.binomial(I_arr_10, D_arr_0p5))  # E: numpy.ndarray[Any, numpy.dtype[numpy.signedinteger[numpy.typing._64Bit]]
reveal_type(def_gen.binomial(I_arr_like_10, D_arr_like_0p5))  # E: numpy.ndarray[Any, numpy.dtype[numpy.signedinteger[numpy.typing._64Bit]]
reveal_type(def_gen.binomial(I_arr_10, D_arr_0p5, size=1))  # E: numpy.ndarray[Any, numpy.dtype[numpy.signedinteger[numpy.typing._64Bit]]
reveal_type(def_gen.binomial(I_arr_like_10, D_arr_like_0p5, size=1))  # E: numpy.ndarray[Any, numpy.dtype[numpy.signedinteger[numpy.typing._64Bit]]

reveal_type(def_gen.negative_binomial(10, 0.5))  # E: int
reveal_type(def_gen.negative_binomial(10, 0.5, size=None))  # E: int
reveal_type(def_gen.negative_binomial(10, 0.5, size=1))  # E: numpy.ndarray[Any, numpy.dtype[numpy.signedinteger[numpy.typing._64Bit]]
reveal_type(def_gen.negative_binomial(I_arr_10, 0.5))  # E: numpy.ndarray[Any, numpy.dtype[numpy.signedinteger[numpy.typing._64Bit]]
reveal_type(def_gen.negative_binomial(10, D_arr_0p5))  # E: numpy.ndarray[Any, numpy.dtype[numpy.signedinteger[numpy.typing._64Bit]]
reveal_type(def_gen.negative_binomial(I_arr_10, 0.5, size=1))  # E: numpy.ndarray[Any, numpy.dtype[numpy.signedinteger[numpy.typing._64Bit]]
reveal_type(def_gen.negative_binomial(10, D_arr_0p5, size=1))  # E: numpy.ndarray[Any, numpy.dtype[numpy.signedinteger[numpy.typing._64Bit]]
reveal_type(def_gen.negative_binomial(I_arr_like_10, 0.5))  # E: numpy.ndarray[Any, numpy.dtype[numpy.signedinteger[numpy.typing._64Bit]]
reveal_type(def_gen.negative_binomial(10, D_arr_like_0p5))  # E: numpy.ndarray[Any, numpy.dtype[numpy.signedinteger[numpy.typing._64Bit]]
reveal_type(def_gen.negative_binomial(I_arr_10, D_arr_0p5))  # E: numpy.ndarray[Any, numpy.dtype[numpy.signedinteger[numpy.typing._64Bit]]
reveal_type(def_gen.negative_binomial(I_arr_like_10, D_arr_like_0p5))  # E: numpy.ndarray[Any, numpy.dtype[numpy.signedinteger[numpy.typing._64Bit]]
reveal_type(def_gen.negative_binomial(I_arr_10, D_arr_0p5, size=1))  # E: numpy.ndarray[Any, numpy.dtype[numpy.signedinteger[numpy.typing._64Bit]]
reveal_type(def_gen.negative_binomial(I_arr_like_10, D_arr_like_0p5, size=1))  # E: numpy.ndarray[Any, numpy.dtype[numpy.signedinteger[numpy.typing._64Bit]]

reveal_type(def_gen.hypergeometric(20, 20, 10))  # E: int
reveal_type(def_gen.hypergeometric(20, 20, 10, size=None))  # E: int
reveal_type(def_gen.hypergeometric(20, 20, 10, size=1))  # E: numpy.ndarray[Any, numpy.dtype[numpy.signedinteger[numpy.typing._64Bit]]
reveal_type(def_gen.hypergeometric(I_arr_20, 20, 10))  # E: numpy.ndarray[Any, numpy.dtype[numpy.signedinteger[numpy.typing._64Bit]]
reveal_type(def_gen.hypergeometric(20, I_arr_20, 10))  # E: numpy.ndarray[Any, numpy.dtype[numpy.signedinteger[numpy.typing._64Bit]]
reveal_type(def_gen.hypergeometric(I_arr_20, 20, I_arr_like_10, size=1))  # E: numpy.ndarray[Any, numpy.dtype[numpy.signedinteger[numpy.typing._64Bit]]
reveal_type(def_gen.hypergeometric(20, I_arr_20, 10, size=1))  # E: numpy.ndarray[Any, numpy.dtype[numpy.signedinteger[numpy.typing._64Bit]]
reveal_type(def_gen.hypergeometric(I_arr_like_20, 20, I_arr_10))  # E: numpy.ndarray[Any, numpy.dtype[numpy.signedinteger[numpy.typing._64Bit]]
reveal_type(def_gen.hypergeometric(20, I_arr_like_20, 10))  # E: numpy.ndarray[Any, numpy.dtype[numpy.signedinteger[numpy.typing._64Bit]]
reveal_type(def_gen.hypergeometric(I_arr_20, I_arr_20, 10))  # E: numpy.ndarray[Any, numpy.dtype[numpy.signedinteger[numpy.typing._64Bit]]
reveal_type(def_gen.hypergeometric(I_arr_like_20, I_arr_like_20, 10))  # E: numpy.ndarray[Any, numpy.dtype[numpy.signedinteger[numpy.typing._64Bit]]
reveal_type(def_gen.hypergeometric(I_arr_20, I_arr_20, I_arr_10, size=1))  # E: numpy.ndarray[Any, numpy.dtype[numpy.signedinteger[numpy.typing._64Bit]]
reveal_type(def_gen.hypergeometric(I_arr_like_20, I_arr_like_20, I_arr_like_10, size=1))  # E: numpy.ndarray[Any, numpy.dtype[numpy.signedinteger[numpy.typing._64Bit]]


reveal_type(def_gen.bit_generator)  # E: BitGenerator

reveal_type(def_gen.bytes(2))  # E: bytes

reveal_type(def_gen.choice(5))  # E: int
reveal_type(def_gen.choice(5, 3))  # E: numpy.ndarray[Any, numpy.dtype[numpy.signedinteger[numpy.typing._64Bit]]
reveal_type(def_gen.choice(5, 3, replace=True))  # E: numpy.ndarray[Any, numpy.dtype[numpy.signedinteger[numpy.typing._64Bit]]
reveal_type(def_gen.choice(5, 3, p=[1 / 5] * 5))  # E: numpy.ndarray[Any, numpy.dtype[numpy.signedinteger[numpy.typing._64Bit]]
reveal_type(def_gen.choice(5, 3, p=[1 / 5] * 5, replace=False))  # E: numpy.ndarray[Any, numpy.dtype[numpy.signedinteger[numpy.typing._64Bit]]

reveal_type(def_gen.choice(["pooh", "rabbit", "piglet", "Christopher"]))  # E: Any
reveal_type(def_gen.choice(["pooh", "rabbit", "piglet", "Christopher"], 3))  # E: numpy.ndarray[Any, Any]
reveal_type(def_gen.choice(["pooh", "rabbit", "piglet", "Christopher"], 3, p=[1 / 4] * 4))  # E: numpy.ndarray[Any, Any]
reveal_type(def_gen.choice(["pooh", "rabbit", "piglet", "Christopher"], 3, replace=True))  # E: numpy.ndarray[Any, Any]
reveal_type(def_gen.choice(["pooh", "rabbit", "piglet", "Christopher"], 3, replace=False, p=np.array([1 / 8, 1 / 8, 1 / 2, 1 / 4])))  # E: numpy.ndarray[Any, Any]

reveal_type(def_gen.dirichlet([0.5, 0.5]))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.dirichlet(np.array([0.5, 0.5])))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.dirichlet(np.array([0.5, 0.5]), size=3))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]

reveal_type(def_gen.multinomial(20, [1 / 6.0] * 6))  # E: numpy.ndarray[Any, numpy.dtype[numpy.signedinteger[numpy.typing._64Bit]]
reveal_type(def_gen.multinomial(20, np.array([0.5, 0.5])))  # E: numpy.ndarray[Any, numpy.dtype[numpy.signedinteger[numpy.typing._64Bit]]
reveal_type(def_gen.multinomial(20, [1 / 6.0] * 6, size=2))  # E: numpy.ndarray[Any, numpy.dtype[numpy.signedinteger[numpy.typing._64Bit]]
reveal_type(def_gen.multinomial([[10], [20]], [1 / 6.0] * 6, size=(2, 2)))  # E: numpy.ndarray[Any, numpy.dtype[numpy.signedinteger[numpy.typing._64Bit]]
reveal_type(def_gen.multinomial(np.array([[10], [20]]), np.array([0.5, 0.5]), size=(2, 2)))  # E: numpy.ndarray[Any, numpy.dtype[numpy.signedinteger[numpy.typing._64Bit]]

reveal_type(def_gen.multivariate_hypergeometric([3, 5, 7], 2))  # E: numpy.ndarray[Any, numpy.dtype[numpy.signedinteger[numpy.typing._64Bit]]
reveal_type(def_gen.multivariate_hypergeometric(np.array([3, 5, 7]), 2))  # E: numpy.ndarray[Any, numpy.dtype[numpy.signedinteger[numpy.typing._64Bit]]
reveal_type(def_gen.multivariate_hypergeometric(np.array([3, 5, 7]), 2, size=4))  # E: numpy.ndarray[Any, numpy.dtype[numpy.signedinteger[numpy.typing._64Bit]]
reveal_type(def_gen.multivariate_hypergeometric(np.array([3, 5, 7]), 2, size=(4, 7)))  # E: numpy.ndarray[Any, numpy.dtype[numpy.signedinteger[numpy.typing._64Bit]]
reveal_type(def_gen.multivariate_hypergeometric([3, 5, 7], 2, method="count"))  # E: numpy.ndarray[Any, numpy.dtype[numpy.signedinteger[numpy.typing._64Bit]]
reveal_type(def_gen.multivariate_hypergeometric(np.array([3, 5, 7]), 2, method="marginals"))  # E: numpy.ndarray[Any, numpy.dtype[numpy.signedinteger[numpy.typing._64Bit]]

reveal_type(def_gen.multivariate_normal([0.0], [[1.0]]))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.multivariate_normal([0.0], np.array([[1.0]])))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.multivariate_normal(np.array([0.0]), [[1.0]]))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]
reveal_type(def_gen.multivariate_normal([0.0], np.array([[1.0]])))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[numpy.typing._64Bit]]

reveal_type(def_gen.permutation(10))  # E: numpy.ndarray[Any, numpy.dtype[numpy.signedinteger[numpy.typing._64Bit]]
reveal_type(def_gen.permutation([1, 2, 3, 4]))  # E: numpy.ndarray[Any, Any]
reveal_type(def_gen.permutation(np.array([1, 2, 3, 4])))  # E: numpy.ndarray[Any, Any]
reveal_type(def_gen.permutation(D_2D, axis=1))  # E: numpy.ndarray[Any, Any]
reveal_type(def_gen.permuted(D_2D))  # E: numpy.ndarray[Any, Any]
reveal_type(def_gen.permuted(D_2D_like))  # E: numpy.ndarray[Any, Any]
reveal_type(def_gen.permuted(D_2D, axis=1))  # E: numpy.ndarray[Any, Any]
reveal_type(def_gen.permuted(D_2D, out=D_2D))  # E: numpy.ndarray[Any, Any]
reveal_type(def_gen.permuted(D_2D_like, out=D_2D))  # E: numpy.ndarray[Any, Any]
reveal_type(def_gen.permuted(D_2D_like, out=D_2D))  # E: numpy.ndarray[Any, Any]
reveal_type(def_gen.permuted(D_2D, axis=1, out=D_2D))  # E: numpy.ndarray[Any, Any]

reveal_type(def_gen.shuffle(np.arange(10)))  # E: None
reveal_type(def_gen.shuffle([1, 2, 3, 4, 5]))  # E: None
reveal_type(def_gen.shuffle(D_2D, axis=1))  # E: None
reveal_type(def_gen.shuffle(D_2D_like, axis=1))  # E: None

reveal_type(np.random.Generator(pcg64))  # E: Generator
reveal_type(def_gen.__str__())  # E: str
reveal_type(def_gen.__repr__())  # E: str
def_gen_state = def_gen.__getstate__()
reveal_type(def_gen_state)  # E: builtins.dict[builtins.str, Any]
reveal_type(def_gen.__setstate__(def_gen_state))  # E: None
