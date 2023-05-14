#ifndef NN2FPGA_QUANTISATION_H_
#define NN2FPGA_QUANTISATION_H_

#include "ap_int.h"
#include "hls_stream.h"

namespace nn2fpga {

template <typename din_t, int SCALE>
din_t quant_act(din_t din) {
  constexpr int SCALE_INV = -1 * SCALE;
  if (SCALE <= 0)
    return (din << SCALE_INV);
  else {
    din_t round = (din >> (SCALE - 1)) & 0x1;
    /* din_t round = 0; */
    return ((din >> SCALE) + round);
  }
}

template <typename din_t, int SCALE, int MASK>
din_t quant_act(din_t din) {
#pragma HLS inline
  constexpr int SCALE_INV = -1 * SCALE;
  din_t val = din;

  constexpr din_t MSB = sizeof(din_t) * 8 - 1;
  const din_t c_mask_tmp = (1 << (MASK + SCALE)) - 1;
  const din_t c_mask_pad = MSB - c_mask_tmp;

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

template <typename din_t, int SCALE, typename dout_t>
dout_t quant_act(din_t din) {
  constexpr dout_t MSB = sizeof(dout_t) * 8 - 1;

  constexpr dout_t MAX_0 = ~(dout_t(0));
  const dout_t MAX_1 = MAX_0 ^ (-1 << MSB);
  const dout_t MAX = (MAX_0 < 0) ? MAX_1 : MAX_0;

  constexpr dout_t MIN_0 = ~(dout_t(0));
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

template <typename din_t, int SCALE, int CLIP, typename dout_t>
dout_t quant_act(din_t din) {
  din_t val = quant_act<din_t, SCALE>(din);
  constexpr dout_t MSB = sizeof(dout_t) * 8 - 1;

  constexpr dout_t MAX = CLIP;

  constexpr dout_t MIN_0 = ~(dout_t)(0);
  constexpr dout_t MIN_1 = (-1 << MSB);
  constexpr dout_t MIN = (MIN_0 < 0) ? MIN_1 : 0;

  if (val > MAX) {
    return MAX;
  }
  if (val < MIN) {
    return MIN;
  }
  return dout_t(val);
}

template <typename din_t, int SCALE, int CLIP, int MASK, typename dout_t>
dout_t quant_act(din_t din) {
  din_t val = quant_act<din_t, SCALE, MASK>(din);

  constexpr dout_t MSB = sizeof(dout_t) * 8 - 1;
  constexpr dout_t MAX = CLIP;

  constexpr dout_t MIN_0 = ~(dout_t)(0);
  /* const dout_t MIN_1 = (-1 << MSB); */
  constexpr dout_t MIN_1 = (-1 * CLIP) - 1;
  constexpr dout_t MIN = (MIN_0 < 0) ? MIN_1 : 0;

  if (val > MAX) {
    return MAX;
  }
  if (val < MIN) {
    return MIN;
  }
  return dout_t(val);
}

} // namespace nn2fpga

#endif // NN2FPGA_QUANTISATION_H_
