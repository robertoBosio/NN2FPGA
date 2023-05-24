#ifndef NN2FPGA_DEBUG_H_
#define NN2FPGA_DEBUG_H_

#ifndef __SYNTHESIS__
/* #define DEBUG */
/* #define DEBUG_ACC */
/* #define DEBUG_LINE */
/* #define DEBUG_POOL */
#endif  // __SYNTHESIS__

#include "hls_stream.h"

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

}  // namespace nn2fpga

#endif  // NN2FPGA_DEBUG_H_
