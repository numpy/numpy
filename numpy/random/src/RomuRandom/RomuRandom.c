
// Romu Random states made into arrays for numpy

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

#include <stdint.h>
#include "romu_random_array.h"

/* === 64 BIT RNG Generators === */

//===== RomuQuad ==================================================================================
//
// More robust than anyone could need, but uses more registers than RomuTrio.
// Est. capacity >= 2^90 bytes. Register pressure = 8 (high). State size = 256 bits.


static inline uint64_t romuquad_next(uint64_t* state){
    uint64_t wp = state[0], xp = state[1], yp = state[2], zp = state[3];
    state[0] = 15241094284759029579u * zp; // a-mult
    state[1] = zp + ROTL(wp,52);           // b-rotl, c-add
    state[2] = yp - xp;                    // d-sub
    state[3] = yp + wp;                    // e-add
    state[3] = ROTL(state[3], 19);            // f-rotl
    return xp;
}

static inline uint64_t romuquad_next64(romuquad_state* state){
    return romuquad_next(state->state);
}

static inline uint32_t romuquad_next32(romuquad_state* state){
    uint64_t next;
    if (state->has_uint32) {
      state->has_uint32 = 0;
      return state->uinteger;
    }
    next = romuquad_next64(state);
    state->has_uint32 = 1;
    state->uinteger = (uint32_t)(next >> 32);
    return (uint32_t)(next & 0xffffffff);
}   



//===== RomuTrio ==================================================================================
//
// Great for general purpose work, including huge jobs.
// Est. capacity = 2^75 bytes. Register pressure = 6. State size = 192 bits.



uint64_t romutrio_next(uint64_t* state) {
   uint64_t xp = state[0] , yp = state[1], zp = state[2];
   state[0] = 15241094284759029579u * zp;
   state[1] = yp - xp;  state[1] = ROTL(state[1], 12);
   state[2] = zp - yp;  state[2] = ROTL(state[2], 44);
   return xp;
}


static inline uint64_t romutrio_next64(romuquad_state* state){
    return romutrio_next(state->state);
}

static inline uint32_t romutrio_next32(romuquad_state* state){
    uint64_t next;
    if (state->has_uint32) {
      state->has_uint32 = 0;
      return state->uinteger;
    }
    next = romutrio_next64(state);
    state->has_uint32 = 1;
    state->uinteger = (uint32_t)(next >> 32);
    return (uint32_t)(next & 0xffffffff);
}  


//===== RomuDuo ==================================================================================
//
// Might be faster than RomuTrio due to using fewer registers, but might struggle with massive jobs.
// Est. capacity = 2^61 bytes. Register pressure = 5. State size = 128 bits.



uint64_t romuduo_next (uint64_t *state) {
   uint64_t xp = state[0];
   state[0] = 15241094284759029579u * state[1];
   state[1] = ROTL(state[1],36) + ROTL(state[1], 15) - xp;
   return xp;
}

static inline uint64_t romuduo_next64(romuduo_state* state){
    return romuduo_next(state->state);
}

static inline uint32_t romuduo_next32(romuduo_state* state){
    uint64_t next;
    if (state->has_uint32) {
      state->has_uint32 = 0;
      return state->uinteger;
    }
    next = romuquad_next64(state);
    state->has_uint32 = 1;
    state->uinteger = (uint32_t)(next >> 32);
    return (uint32_t)(next & 0xffffffff);
}   

//===== RomuDuoJr ================================================================================
//
// The fastest generator using 64-bit arith., but not suited for huge jobs.
// Est. capacity = 2^51 bytes. Register pressure = 4. State size = 128 bits.

// typedef struct s_romuduojr_state{
//     uint64_t state[2];
//     int has_uint32;
//     uint32_t uinteger;
// } romuduojr_state;


uint64_t romuduojr_next (uint64_t *state) {
   uint64_t xp = state[0];
   state[0] = 15241094284759029579u * state[1];
   state[1] = state[0] - xp;  state[1] = ROTL(state[1], 27);
   return xp;
}

static inline uint64_t romuduojr_next64(romuduojr_state* state){
    return romuduojr_next(state->state);
}

static inline uint32_t romuduojr_next32(romuduo_state* state){
    uint64_t next;
    if (state->has_uint32) {
      state->has_uint32 = 0;
      return state->uinteger;
    }
    next = romuduojr_next64(state);
    state->has_uint32 = 1;
    state->uinteger = (uint32_t)(next >> 32);
    return (uint32_t)(next & 0xffffffff);
}   


/* === 32 Bit RNG GENERATORS === */


//===== RomuQuad32 ================================================================================
//
// 32-bit arithmetic: Good for general purpose use.
// Est. capacity >= 2^62 bytes. Register pressure = 7. State size = 128 bits.


typedef struct s_romuquad32_state{
    uint32_t state[4];
} romuquad32_state;

uint32_t romuquad32_next(uint32_t *state) {
    uint32_t wp = state[0], xp = state[1], yp = state[2], zp = state[3];
    state[0] = 3323815723u * zp;  // a-mult
    state[1] = zp + ROTL(wp,26);  // b-rotl, c-add
    state[2] = yp - xp;           // d-sub
    state[3] = yp + wp;           // e-add
    state[3] = ROTL(state[3],9);    // f-rotl
    return xp;
}

static inline uint64_t romuquad32_next64(romuquad32_state *state) { 
    return (uint64_t)romuquad32_next(state) << 32 | romuquad32_next(state); 
}

static inline uint32_t romuquad32_next32(romuquad32_state * state) { 
    return romuquad32_next(state->state); 
}

//===== RomuTrio32 ===============================================================================
//
// 32-bit arithmetic: Good for general purpose use, except for huge jobs.
// Est. capacity >= 2^53 bytes. Register pressure = 5. State size = 96 bits.


uint32_t romutrio32_next (uint32_t *state) {
    uint32_t xp = state[0], yp = state[1], zp = state[2];
    state[0] = 3323815723u * zp;
    state[1] = yp - xp; state[1] = ROTL(state[1],6);
    state[2] = zp - yp; state[2] = ROTL(state[2],22);
    return xp;
 }

static inline uint64_t romutrio32_next64(romutrio32_state *state) { 
    return (uint64_t)romutrio32_next(state) << 32 | romutrio32_next(state); 
} 
static inline uint32_t romutrio32_next32(romutrio32_state * state) { 
    return romutrio32_next(state->state); 
}
