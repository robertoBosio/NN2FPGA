#ifndef NN2FPGA_DEBUG_H_
#define NN2FPGA_DEBUG_H_

#ifndef __SYNTHESIS__
// #define DEBUG
// #define DEBUG_CONV
// #define DEBUG_RES
// #define DEBUG_INPUT
// #define DEBUG_ADD
// #define DEBUG_BANDWIDTH
// #define DEBUG_FORWARD
// #define DEBUG_WEIGHTS
// #define DEBUG_ACT
// #define DEBUG_ACC
// #define DEBUG_LINE
// #define DEBUG_POOL
// #define DEBUG_WEIGHTS
// #define DEBUG_LAST
#endif  // __SYNTHESIS__

#include "hls_stream.h"
#include "ap_int.h"

namespace nn2fpga {

template <typename T>
void empty_stream(hls::stream<T>& dinStream) {
#pragma HLS inline
  /* This handles padding aware inputs */

#ifndef __SYNTHESIS__
  unsigned s_left = 0;
  while (!dinStream.empty()) {
    dinStream.read();
    s_left++;
  }
#ifdef DEBUG
  std::cout << "LEFT: " << s_left << std::endl;
#endif  // DEBUG
#endif  // __SYNTHESIS__
}

template <
  typename t_weights_stream,
  typename t_weights_st,
  int c_weights_dim
> void dma_emulator (
  const t_weights_st c_weights[c_weights_dim],
  hls::stream<t_weights_stream> c_weights_stream[1]
) {

  t_weights_stream s_data;
  for (auto i = 0; i < c_weights_dim; i++) {
    s_data.data = c_weights[i];
    s_data.last = (i == (c_weights_dim - 1));
    c_weights_stream[0].write(s_data);
  }

};


}  // namespace nn2fpga

#endif  // NN2FPGA_DEBUG_H_
