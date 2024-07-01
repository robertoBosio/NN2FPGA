#ifndef NN2FPGA_QUANTISATION_H_
#define NN2FPGA_QUANTISATION_H_

#include "ap_int.h"
#include "hls_stream.h"

namespace nn2fpga {

// Scale a value by a SCALE factor which is a power of 2.
template <typename din_t, int SCALE>
din_t quant_act(din_t din) {
  const int SCALE_INV = -1 * SCALE;
  if (SCALE <= 0) // Left shifting.
    return (din << SCALE_INV);
  else { // Right shifting and LSB rounding.
    din_t round = (din >> (SCALE - 1)) & 0x1;
    return ((din >> SCALE) + round);
  }
}


// Scale a value with a SCALE factor, with additional MASK factor (to be 
// implemented in the future).
template <typename din_t, int SCALE, int MASK>
din_t quant_act(din_t din) {
#pragma HLS inline
  const int SCALE_INV = -1 * SCALE;
  din_t val = din;

  const din_t MSB = sizeof(din_t) * 8 - 1;
  const din_t c_mask_tmp = (1 << (MASK + SCALE)) - 1;
  const din_t c_mask_pad = MSB - c_mask_tmp;

    // print result for debug
  if (SCALE <= 0)
    return (val << SCALE_INV);
  else {
    /* if (MASK > 0) */
    /*   s_data = (din_t)(s_data & c_mask_pad); */
    din_t round = (val >> (SCALE - 1)) & 0x1;
    /* return (din_t)(((s_data >> SCALE) + round) & MASK); */
    return ((val >> SCALE) + round);
  }
}


// Scale a factor by a factor SCALE (power of 2) and clipping it on a different
// bitwidth provided by dout_t.
template <typename din_t, typename dout_t, int SCALE>
dout_t quant_act(din_t din) {
  const dout_t MSB = sizeof(dout_t) * 8 - 1;

  const dout_t MAX_0 = ~(dout_t(0));
  const dout_t MAX_1 = MAX_0 ^ (-1 << MSB); // How does a negative value shift 
                                            // work?
  const dout_t MAX = (MAX_0 < 0) ? MAX_1 : MAX_0;

  const dout_t MIN_0 = ~(dout_t(0));
  const dout_t MIN_1 = (-1 << MSB);
  const dout_t MIN = (MIN_0 < 0) ? MIN_1 : 0;

  din_t val = quant_act<din_t, SCALE, MAX_0>(din);

  if (val > MAX) {
    return MAX;
  }
  if (val < MIN) {
    return MIN;
  }
  return dout_t(val);
}


// Scaling and clipping a value on a different bitwidth provided by dout_t.
template <typename din_t, typename dout_t, int SCALE, int CLIP>
dout_t quant_act(din_t din) {
  din_t val = quant_act<din_t, SCALE>(din);
  const dout_t MSB = sizeof(dout_t) * 8 - 1;

  const dout_t MAX = CLIP;

  const dout_t MIN_0 = ~(dout_t)(0);
  const dout_t MIN_1 = (-1 << MSB);
  const dout_t MIN = (MIN_0 < 0) ? MIN_1 : 0;

  if (val > MAX) {
    return MAX;
  }
  if (val < MIN) {
    return MIN;
  }
  return dout_t(val);
}
 

// Scaling, clipping on dout_t, masking.
template <typename din_t, typename dout_t, int SCALE, int CLIP, int MASK>
dout_t quant_act(din_t din) {
  din_t val = quant_act<din_t, SCALE, MASK>(din);

  const dout_t MSB = sizeof(dout_t) * 8 - 1;
  const dout_t MAX = CLIP;

  const dout_t MIN_0 = ~(dout_t)(0);
  /* const dout_t MIN_1 = (-1 << MSB); */
  const dout_t MIN_1 = (-1 * CLIP) - 1;
  const dout_t MIN = (MIN_0 < 0) ? MIN_1 : 0;

  if (val > MAX) {
    return MAX;
  }
  if (val < MIN) {
    return MIN;
  }
  return dout_t(val);
}

}  // namespace nn2fpga

#endif  // NN2FPGA_QUANTISATION_H_
