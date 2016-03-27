//          Copyright Boston University SESA Group 2013 - 2015.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE_1_0.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)
#ifndef APPS_PCG_BASIC_H_
#define APPS_PCG_BASIC_H_

#include <cinttypes>

//#define PCG32_INITIALIZER   { 0x853c49e6748fea9bULL, 0xda3e39cb94b95bdbULL }

class pcg_basic {
public:
    pcg_basic(uint64_t initstate, uint64_t initseq);
    
    uint32_t pcg_random();

    uint32_t pcg_boundrand(uint32_t bound);
    
private:
    uint64_t state;
    uint64_t inc;
};

#endif // APPS_PCG_BASIC_H_
