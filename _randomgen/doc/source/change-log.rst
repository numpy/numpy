Change Log
----------
v1.15.2
=======
- Fixed a bug that affected :class:`~randomgen.dsfmt.DSFMT` when calling
  :func:`~randomgen.dsfmt.DSFMT.jump` or :func:`~randomgen.dsfmt.DSFMT.seed`
  that failed to reset the buffer.  This resulted in upto 381 values from the
  previous state being used before the buffer was refilled at the new state.
- Fixed bugs in :class:`~randomgen.xoshiro512starstar.Xoshiro512StarStar`
  and :class:`~randomgen.xorshift1024.Xorshift1024` where the fallback
  entropy initialization used too few bytes. This bug is unlikely to be
  encountered since this path is only encountered if the system random
  number generator fails.

v1.15.1
=======
- Added Xoshiro256** and Xoshiro512**, the preferred generators of this class.
- Fixed bug in `jump` method of Random123 generators which did nto specify a default value.
- Added support for generating bounded uniform integers using Lemire's method.
- Synchronized with upstream changes, which requires moving the minimum supported NumPy to 1.13.

v1.15
=====
- Synced empty choice changes
- Synced upstream docstring changes
- Synced upstream changes in permutation
- Synced upstream doc fixes
- Added absolute_import to avoid import noise on Python 2.7
- Add legacy generator which allows NumPy replication
- Improve type handling of integers
- Switch to array-fillers for 0 parameter distribution to improve performance
- Small changes to build on manylinux
- Build wheels using multibuild
