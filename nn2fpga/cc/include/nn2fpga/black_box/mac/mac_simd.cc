#include "mac_simd.h"

void mac_simd(ap_int<27> a, ap_int<27> d, ap_int<18> b, ap_uint<48> &p) {
#pragma HLS inline off

  p += (a + d) * b;

}
