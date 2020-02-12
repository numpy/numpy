Enable multi-platform SIMD compiler optimizations

The new NumPy's infrastructure upgrade provides multi-platform and cross-architecture build options to effectively control of compiler optimizations that are mainly related to CPU features

## Build options

`--cpu-baseline` minimal set of required optimizations, default `"min"`

`--cpu-dipsatch` dispatched set of additional optimizations, default `"max -xop -fma4"`

Optimization names can be CPU features or group of features that gather several features or special options perform a series of procedures.

The following tables show the current supported optimizations sorted from the lowest to the highest interest.

#### `X86` - CPU feature names:

| Name       | Implies                                                      |
| ---------- | ------------------------------------------------------------ |
| `SSE`      | `NONE`                                                       |
| `SSE2`     | `SSE`                                                        |
| `SSE3`     | `SSE` `SSE2`                                                 |
| `SSSE3`    | `SSE` `SSE2` `SSE3`                                          |
| `SSE41`    | `SSE` `SSE2` `SSE3` `SSSE3`                                  |
| `POPCNT`   | `SSE` `SSE2` `SSE3` `SSSE3` `SSE41`                          |
| `SSE42`    | `SSE` `SSE2` `SSE3` `SSSE3` `SSE41` `POPCNT`                 |
| `AVX`      | `SSE` `SSE2` `SSE3` `SSSE3` `SSE41` `POPCNT` `SSE42`         |
| `F16C`     | `SSE` `SSE2` `SSE3` `SSSE3` `SSE41` `POPCNT` `SSE42` `AVX`   |
| `XOP`      | `SSE` `SSE2` `SSE3` `SSSE3` `SSE41` `POPCNT` `SSE42` `AVX`   |
| `FMA4`     | `SSE` `SSE2` `SSE3` `SSSE3` `SSE41` `POPCNT` `SSE42` `AVX`   |
| `FMA3`     | `SSE` `SSE2` `SSE3` `SSSE3` `SSE41` `POPCNT` `SSE42` `AVX`   |
| `AVX2`     | `SSE` `SSE2` `SSE3` `SSSE3` `SSE41` `POPCNT` `SSE42` `AVX` `F16C` `FMA3` |
| `AVX512F`  | `SSE` `SSE2` `SSE3` `SSSE3` `SSE41` `POPCNT` `SSE42` `AVX` `F16C` `FMA3` `AVX2` |
| `AVX512CD` | `SSE` `SSE2` `SSE3` `SSSE3` `SSE41` `POPCNT` `SSE42` `AVX` `F16C` `FMA3` `AVX2` `AVX512F` |

#### `X86` - Group names:

| Name         | Gather                                           | Implies                                                      |
| ------------ | ------------------------------------------------ | ------------------------------------------------------------ |
| `AVX512_KNL` | `AVX512ER` `AVX512PF`                            | `SSE*` ` POPCNT` `AVX` `F16C` `FMA3` `AVX2` `AVX512F` `AVX512CD` |
| `AVX512_KNM` | `AVX5124FMAPS` ` AVX5124VNNIW` `AVX512VPOPCNTDQ` | `SSE*` ` POPCNT` `AVX` `F16C` `FMA3` `AVX2` `AVX512F` `AVX512CD` ``AVX512_KNL`` |
| `AVX512_SKX` | `AVX512VL` `AVX512BW` `AVX512DQ`                 | `SSE*` ` POPCNT` `AVX` `F16C` `FMA3` `AVX2` `AVX512F` `AVX512CD` |
| `AVX512_CLX` | `AVX512VNNI`                                     | `SSE*` ` POPCNT` `AVX` `F16C` `FMA3` `AVX2` `AVX512F` `AVX512CD` ``AVX512_SKX`` |
| `AVX512_CNL` | `AVX512IFM` `AVX512VBMI`                         | `SSE*` ` POPCNT` `AVX` `F16C` `FMA3` `AVX2` `AVX512F` `AVX512CD` ``AVX512_SKX`` |
| `AVX512_ICL` | `AVX512VBMI2` `AVX512BITALG`  `AVX512VPOPCNTDQ`  | `SSE*` ` POPCNT` `AVX` `F16C` `FMA3` `AVX2` `AVX512F` `AVX512CD` ``AVX512_SKX`` `AVX512_CLX` ``AVX512_CNL`` |

#### `IBM/POWER` - CPU feature names:

| Name   | Implies      |
| ------ | ------------ |
| `VSX`  | `NONE`       |
| `VSX2` | `VSX`        |
| `VSX3` | `VSX` `VSX2` |

#### `ARM` - CPU feature names:

| Name         | Implies                                             |
| ------------ | --------------------------------------------------- |
| `NEON`       | `NONE`                                              |
| `NEON_FP16`  | `NEON`                                              |
| `NEON_VFPV4` | `NEON` ``NEON_FP16``                                |
| `ASIMD`      | `NEON` ``NEON_FP16`` `NEON_VFPV4`                   |
| `ASIMDHP`    | `NEON` ``NEON_FP16`` `NEON_VFPV4` `ASIMD`           |
| `ASIMDDP`    | `NEON` ``NEON_FP16`` `NEON_VFPV4` `ASIMD`           |
| `ASIMDFHM`   | `NEON` ``NEON_FP16`` `NEON_VFPV4` `ASIMD` `ASIMDHP` |

#### Special options:

`NONE` : enable no features

`NATIVE`:  fetch all CPU features and groups the current machine supports, this operation is based on the compiler flags (`-march=native, -xHost, /QxHost`)

`MIN`:  the safest features for wide range of users platforms, explained as following:

| For Arch                         | Returns                                 |
| -------------------------------- | --------------------------------------- |
| `x86`                            | `SSE` `SSE2`                            |
| `x86` `64-bit mode`              | `SSE` `SSE2` `SSE3`                     |
| `IBM/POWER` `big-endian mode`    | `NONE`                                  |
| `IBM/POWER` `little-endian mode` | `VSX` `VSX2`                            |
| `ARMHF`                          | `NONE`                                  |
| `ARM64` `AARCH64`                | `NEON` `NEON_FP16` `NEON_VFPV4` `ASIMD` |

`MAX:` fetch all CPU features and groups that supported by current platform and compiler build.

`Operators -/+: ` add or sub features and options.

#### Special cases:

#### 

#### Behaviors and Errors 

#### 

#### Usage and Examples:

#### 

#### Report and Trace:

#### 



## Understanding CPU Dispatching

### The baseline:

#### 

#### Dispatcher:

#### 

#### Groups and Policies:

#### 

#### Examples:

#### 

#### Report and Trace:

#### 



## References 

#### 
