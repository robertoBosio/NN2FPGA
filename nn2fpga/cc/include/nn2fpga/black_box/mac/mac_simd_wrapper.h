#ifndef MAC_SIMD_WRAPPER_H_
#define MAC_SIMD_WRAPPER_H_

#include "ap_int.h"
#include "nn2fpga/black_box/mac/mac_simd.h"

void mac_simd_wrapper(ap_int<27> a, ap_int<27> d, ap_int<18> b, ap_uint<48> &p) {
    // #pragma HLS inline
    #pragma HLS interface mode=ap_ctrl_none port=return

    mac_simd(a, d, b, p);
}

#endif
