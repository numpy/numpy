.. _upgrading-pcg64:

.. currentmodule:: numpy.random

Upgrading ``PCG64`` with ``PCG64DXSM``
======================================

Uses of the `PCG64` `BitGenerator` in a massively-parallel context have been
shown to have statistical weaknesses that were not apparent at the first
release in numpy 1.17. Most users will never observe this weakness and are
safe to continue to use `PCG64`. We have introduced a new `PCG64DXSM`
`BitGenerator` that will eventually become the new default `BitGenerator`
implementation used by `default_rng` in future releases. `PCG64DXSM` solves
the statistical weakness while preserving the performance and the features of
`PCG64`.

Does this affect me?
--------------------

If you

  1. only use a single `Generator` instance,
  2. only use `RandomState` or the functions in `numpy.random`,
  3. only use the `PCG64.jumped` method to generate parallel streams,
  4. explicitly use a `BitGenerator` other than `PCG64`,

then this weakness does not affect you at all. Carry on.

If you use moderate numbers of parallel streams created with `default_rng` or
`SeedSequence.spawn`, in the 1000s, then the chance of observing this weakness
is negligibly small. You can continue to use `PCG64` comfortably.

If you use very large numbers of parallel streams, in the millions, and draw
large amounts of numbers from each, then the chance of observing this weakness
can become non-negligible, if still small. An example of such a use case would
be a very large distributed reinforcement learning problem with millions of
long Monte Carlo playouts each generating billions of random number draws. Such
use cases should consider using `PCG64DXSM` explicitly or another
modern `BitGenerator` like `SFC64` or `Philox`, but it is unlikely that any
old results you may have calculated are invalid. In any case, the weakness is
a kind of `Birthday Paradox <https://en.wikipedia.org/wiki/Birthday_problem>`_
collision. That is, a single pair of parallel streams out of the millions,
considered together, might fail a stringent set of statistical tests of
randomness. The remaining millions of streams would all be perfectly fine, and
the effect of the bad pair in the whole calculation is very likely to be
swamped by the remaining streams in most applications.

.. _upgrading-pcg64-details:

Technical Details
-----------------

Like many PRNG algorithms, `PCG64` is constructed from a transition function,
which advances a 128-bit state, and an output function, that mixes the 128-bit
state into a 64-bit integer to be output. One of the guiding design principles
of the PCG family of PRNGs is to balance the computational cost (and
pseudorandomness strength) between the transition function and the output
function. The transition function is a 128-bit linear congruential generator
(LCG), which consists of multiplying the 128-bit state with a fixed
multiplication constant and then adding a user-chosen increment, in 128-bit
modular arithmetic. LCGs are well-analyzed PRNGs with known weaknesses, though
128-bit LCGs are large enough to pass stringent statistical tests on their own,
with only the trivial output function. The output function of `PCG64` is
intended to patch up some of those known weaknesses by doing "just enough"
scrambling of the bits to assist in the statistical properties without adding
too much computational cost.

One of these known weaknesses is that advancing the state of the LCG by steps
numbering a power of two (``bg.advance(2**N)``) will leave the lower ``N`` bits
identical to the state that was just left. For a single stream drawn from
sequentially, this is of little consequence. The remaining :math:`128-N` bits provide
plenty of pseudorandomness that will be mixed in for any practical ``N`` that can
be observed in a single stream, which is why one does not need to worry about
this if you only use a single stream in your application. Similarly, the
`PCG64.jumped` method uses a carefully chosen number of steps to avoid creating
these collisions. However, once you start creating "randomly-initialized"
parallel streams, either using OS entropy by calling `default_rng` repeatedly
or using `SeedSequence.spawn`, then we need to consider how many lower bits
need to "collide" in order to create a bad pair of streams, and then evaluate
the probability of creating such a collision.
`Empirically <https://github.com/numpy/numpy/issues/16313>`_, it has been
determined that if one shares the lower 58 bits of state and shares an
increment, then the pair of streams, when interleaved, will fail 
`PractRand <http://pracrand.sourceforge.net/>`_ in
a reasonable amount of time, after drawing a few gigabytes of data. Following
the standard Birthday Paradox calculations for a collision of 58 bits, we can
see that we can create :math:`2^{29}`, or about half a billion, streams which is when
the probability of such a collision becomes high. Half a billion streams is
quite high, and the amount of data each stream needs to draw before the
statistical correlations become apparent to even the strict ``PractRand`` tests
is in the gigabytes. But this is on the horizon for very large applications
like distributed reinforcement learning. There are reasons to expect that even
in these applications a collision probably will not have a practical effect in
the total result, since the statistical problem is constrained to just the
colliding pair.

Now, let us consider the case when the increment is not constrained to be the
same. Our implementation of `PCG64` seeds both the state and the increment;
that is, two calls to `default_rng` (almost certainly) have different states
and increments. Upon our first release, we believed that having the seeded
increment would provide a certain amount of extra protection, that one would
have to be "close" in both the state space and increment space in order to
observe correlations (``PractRand`` failures) in a pair of streams. If that were
true, then the "bottleneck" for collisions would be the 128-bit entropy pool
size inside of `SeedSequence` (and 128-bit collisions are in the
"preposterously unlikely" category). Unfortunately, this is not true.

One of the known properties of an LCG is that different increments create
*distinct* streams, but with a known relationship. Each LCG has an orbit that
traverses all :math:`2^{128}` different 128-bit states. Two LCGs with different
increments are related in that one can "rotate" the orbit of the first LCG
(advance it by a number of steps that we can compute from the two increments)
such that then both LCGs will always then have the same state, up to an
additive constant and maybe an inversion of the bits. If you then iterate both
streams in lockstep, then the states will *always* remain related by that same
additive constant (and the inversion, if present). Recall that `PCG64` is
constructed from both a transition function (the LCG) and an output function.
It was expected that the scrambling effect of the output function would have
been strong enough to make the distinct streams practically independent (i.e.
"passing the ``PractRand`` tests") unless the two increments were
pathologically related to each other (e.g. 1 and 3). The output function XSL-RR
of the then-standard PCG algorithm that we implemented in `PCG64` turns out to
be too weak to cover up for the 58-bit collision of the underlying LCG that we
described above. For any given pair of increments, the size of the "colliding"
space of states is the same, so for this weakness, the extra distinctness
provided by the increments does not translate into extra protection from
statistical correlations that ``PractRand`` can detect.

Fortunately, strengthening the output function is able to correct this weakness
and *does* turn the extra distinctness provided by differing increments into
additional protection from these low-bit collisions. To the `PCG author's
credit <https://github.com/numpy/numpy/issues/13635#issuecomment-506088698>`_,
she had developed a stronger output function in response to related discussions
during the long birth of the new `BitGenerator` system. We NumPy developers
chose to be "conservative" and use the XSL-RR variant that had undergone
a longer period of testing at that time. The DXSM output function adopts
a "xorshift-multiply" construction used in strong integer hashes that has much
better avalanche properties than the XSL-RR output function. While there are
"pathological" pairs of increments that induce "bad" additive constants that
relate the two streams, the vast majority of pairs induce "good" additive
constants that make the merely-distinct streams of LCG states into
practically-independent output streams. Indeed, now the claim we once made
about `PCG64` is actually true of `PCG64DXSM`: collisions are possible, but
both streams have to simultaneously be both "close" in the 128 bit state space
*and* "close" in the 127-bit increment space, so that would be less likely than
the negligible chance of colliding in the 128-bit internal `SeedSequence` pool.
The DXSM output function is more computationally intensive than XSL-RR, but
some optimizations in the LCG more than make up for the performance hit on most
machines, so `PCG64DXSM` is a good, safe upgrade. There are, of course, an
infinite number of stronger output functions that one could consider, but most
will have a greater computational cost, and the DXSM output function has now
received many CPU cycles of testing via ``PractRand`` at this time.
