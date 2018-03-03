# TODO
0. Use inheritance to simplify CorePRNG structure. The natural base is 
   xoroshiro128.
1. Add PCG64
2. Add dSFMT
3. Add xorshift2014
4. Augment state to include has_gauss and gauss
5. Augment state to have binomial structure
6. Port over 0 parameter distributions
   * standard exponential float
   * standard exponential ziggurat
   * standard exponential ziggurat float   
   * standard normal
   * standard normal float
   * standard normal ziggurat
   * standard normal ziggurat float
   * standard gamma
   * standard gamma float
7. Remove SplitMix64 as an external generator
8. Restore ability to use `out` in core distributions
9. Add correct carry for ThreeFry to allow full set of counters.  Important when implmenting jump

