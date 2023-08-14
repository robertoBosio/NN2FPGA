#include "mac_simd.h"

ap_uint<48> mac_simd(ap_int<27> A, ap_int<18> B, ap_uint<48> C, ap_int<27> D) {
#pragma HLS inline off

  return C + (A + D) * B;

}
