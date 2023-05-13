#ifndef NN2FPGA_ACTIVATION_STREAMS_H_
#define NN2FPGA_ACTIVATION_STREAMS_H_

// Vitis HLS dependencies.
#include <ap_int.h>
#include <hls_stream.h>

namespace nn2fpga {

template <typename T>
T relu_op(T val) {
#pragma HLS inline
  return val > 0 ? val : 0;
}

template <typename din_t, typename dout_t, unsigned ICH, unsigned IH,
          unsigned IW>
void relu_streams(hls::stream<din_t>& dinStream,
                  hls::stream<dout_t>& doutStream) {
  if (dinStream.empty()) return;

  din_t din;
  dout_t dout;

  constexpr unsigned ISZ = ICH * IH * IW;

  for (int i = 0; i < ISZ; i++) {
    din << dinStream.read();
    dout = din > 0 ? din : 0;
    doutStream << dout;
  }
}

}  // namespace nn2fpga

#endif  // NN2FPGA_ACTIVATION_STREAMS_H_
