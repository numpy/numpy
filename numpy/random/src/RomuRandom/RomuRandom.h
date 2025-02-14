
// Romu Random states were rewritten into arrays for use with numpy

// Romu Pseudorandom Number Generators
//
// Copyright 2020 Mark A. Overton
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// 
//     http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// ------------------------------------------------------------------------------------------------
//
// Website: romu-random.org
// Paper:   http://arxiv.org/abs/2002.11331


#ifndef __ROMURANDOM_H__
#define __ROMURANDOM_H__

#include <stdint.h>


#ifdef __cplusplus
extern "C" {
#endif /*__cplusplus */


typedef struct s_romuquad_state{
    uint64_t state[4];
    int has_uint32;
    uint32_t uinteger;
} romuquad_state;

static inline uint64_t romuquad_next(uint64_t* state);
static inline uint64_t romuquad_next64(romuquad_state * state);
static inline uint32_t romuquad_next32(romuquad_state * state);

typedef struct s_romutrio_state{
    uint64_t state[3];
    int has_uint32;
    uint32_t uinteger;
} romutrio_state;

static inline uint64_t romutrio_next(uint64_t* state);
static inline uint64_t romutrio_next64(romutrio_state * state);
static inline uint32_t romutrio_next32(romutrio_state * state);

typedef struct s_romuduo_state{
    uint64_t state[2];
    int has_uint32;
    uint32_t uinteger;
} romuduo_state;

static inline uint64_t romuduo_next(uint64_t* state);
static inline uint64_t romuduo_next64(romuduo_state * state);
static inline uint32_t romuduo_next32(romuduo_state * state);

typedef struct s_romuduojr_state{
    uint64_t state[2];
    int has_uint32;
    uint32_t uinteger;
} romuduojr_state;

static inline uint64_t romuduojr_next(uint64_t* state);
static inline uint64_t romuduojr_next64(romuduojr_state * state);
static inline uint32_t romuduojr_next32(romuduojr_state * state);


typedef struct s_romuquad32_state{
    uint32_t state[4];
} romuquad32_state;

static inline uint32_t romuquad32_next(uint32_t* state);
static inline uint64_t romuquad32_next64(romuquad32_state * state);
static inline uint32_t romuquad32_next32(romuquad32_state * state);

typedef struct s_romutrio32_state{
    uint32_t state[3];
} romutrio32_state;

static inline uint32_t romutrio32_next(uint32_t* state);
static inline uint64_t romutrio32_next64(romutrio32_state * state);
static inline uint32_t romutrio32_next32(romutrio32_state * state);

#ifdef __cplusplus
}
#endif /*__cplusplus */

#endif /* __ROMURANDOM_H__ */
