//          Copyright Boston University SESA Group 2013 - 2015.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE_1_0.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)
#include "pcg_basic.h"

// This is *IMPORTANT*, it allows the messenger to resolve remote HandleFaults
//EBBRT_PUBLISH_TYPE(, pcg_basic);

//using namespace ebbrt;

pcg_basic::pcg_basic(uint64_t initstate, uint64_t initseq) {
  state = 0U;
  inc = (initseq << 1u) | 1u;
  pcg_random();
  state += initstate;
  pcg_random();
}

uint32_t pcg_basic::pcg_random() {
  uint64_t oldstate = state;
  state = oldstate * 6364136223846793005ULL + inc;
  uint32_t xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
  uint32_t rot = oldstate >> 59u;
  return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

uint32_t pcg_basic::pcg_boundrand(uint32_t bound) {
  uint32_t threshold = -bound % bound;

  for (;;) {
    uint32_t r = pcg_random();
    if (r >= threshold)
      return r % bound;
  }
}
