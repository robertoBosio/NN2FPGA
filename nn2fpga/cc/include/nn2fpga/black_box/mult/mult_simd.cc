#include "mult_simd.h"

void mult_simd(int8_t a, int8_t d, int8_t b, int16_t& ab, int16_t& bd) {
#pragma HLS inline off
  ab = a * b;
  bd = b * d;
}
