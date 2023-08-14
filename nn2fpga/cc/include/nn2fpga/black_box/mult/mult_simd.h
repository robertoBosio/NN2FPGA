#ifndef MULT_SIMD_H_
#define MULT_SIMD_H_

#include <cstdint>

typedef int8_t mult_simd_din_t;
typedef int16_t mult_simd_dout_t;

void mult_simd(int8_t a, int8_t d, int8_t b, int16_t& ab, int16_t& bd);

#endif  // MUL_SIMD_H_
