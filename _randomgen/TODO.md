# TODO
2. Add dSFMT
5. Augment state to have binomial structure
6. Port over 0 parameter distributions
   * standard exponential ziggurat float   
   * standard normal ziggurat
   * standard normal ziggurat float
7. Remove SplitMix64 as an external generator
8. Restore ability to use `out` in core distributions
10. Seed/Inc for PCG64
11. Advance/Jump for PCG64
12. Key/Counter for ThreeFry

## Done
1. Add PCG64
3. Add xorshift2014
4. Augment state to include has_gauss and gauss
6. Port over 0 parameter distributions
   * standard exponential ziggurat
   * standard exponential float
   * standard normal
   * standard normal float
   * standard gamma - Not implement: This is a 1 param
   * standard gamma float - Not implement: This is a 1 param

9. Add correct carry for ThreeFry to allow full set of counters.  Important when implemeting jump
0. NOT IMPLEMENTABLE due to limits on inheritance in Cython: Use inheritance to simplify CorePRNG structure. The natural base is 
   xoroshiro128.
